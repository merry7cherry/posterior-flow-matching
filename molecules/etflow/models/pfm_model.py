"""Task-native PFM using the ETFlow molecular backbone."""

import math
import torch
from torch import nn
from .model import MolecularBackbone
from . import pfm_probability as prob
from .pfm_sampling import sample as integrate


class PosteriorFlow(MolecularBackbone):
    def __init__(
        self,
        source_scale=1.0,
        logvar_min=-4.0,
        logvar_max=4.0,
        logvar_init=0.0,
        warmup_fraction=0.01,
        **kwargs,
    ):
        if kwargs.get("prior_type", "gaussian") != "gaussian" or kwargs.get("sigma", 0) != 0:
            raise ValueError("PFM requires independent Gaussian source and zero bridge noise")
        kwargs.update(prior_type="gaussian", sigma=0.0)
        super().__init__(**kwargs)
        if not source_scale > 0 or not logvar_min < logvar_init < logvar_max:
            raise ValueError("Invalid source scale or variance initialization")
        self.source_scale, self.logvar_min, self.logvar_max = source_scale, logvar_min, logvar_max
        self.warmup_fraction = warmup_fraction
        width = self.network.representation_model.hidden_channels
        self.network.variance_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.network.variance_head[-1].weight)
        fraction = (logvar_init - logvar_min) / (logvar_max - logvar_min)
        nn.init.constant_(self.network.variance_head[-1].bias, math.log(fraction / (1 - fraction)))

    def predict_posterior(
        self, z, t, pos, bond_index, batch, node_attr=None, edge_attr=None
    ):
        velocity, raw = super().forward(z, t, pos, bond_index, edge_attr, node_attr, batch)
        with torch.autocast(device_type=pos.device.type, enabled=False):
            lv = prob.bounded_logvar(raw.float(), self.logvar_min, self.logvar_max)
            return prob.center(velocity.float(), batch), lv

    def objective(self, velocity, target, logvar, batch):
        return prob.projected_nll(velocity - target, logvar, batch).mean()

    def generic_step(self, data, batch_idx, stage):
        batch = data.batch
        num_mols = int(batch.max()) + 1
        x1 = prob.center(data.pos.float(), batch)
        if stage == "val":
            gen = torch.Generator(device=x1.device).manual_seed(100000 + batch_idx)
        else:
            gen = None
        x0 = prob.center(
            torch.randn(x1.shape, device=x1.device, generator=gen) * self.source_scale, batch
        )
        t = torch.rand((num_mols, 1), device=x1.device, generator=gen) * 0.9998 + 0.0001
        xt = (1 - t[batch]) * x0 + t[batch] * x1
        v, lv = self.predict_posterior(
            data.atomic_numbers, t, xt, data.edge_index, batch, data.node_attr, data.edge_attr,
        )
        with torch.autocast(device_type=x1.device.type, enabled=False):
            loss = self.objective(v, x1 - x0, lv, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite PFM NLL")
        self.log(f"{stage}/loss", loss, batch_size=num_mols, sync_dist=True)
        self.log(
            f"{stage}/velocity_mse",
            (v - (x1 - x0)).square().mean(),
            batch_size=num_mols,
            sync_dist=True,
        )
        self.log(f"{stage}/logvar", lv.mean(), batch_size=num_mols, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.opt_betas, weight_decay=self.weight_decay
        )
        total = int(self.trainer.estimated_stepping_batches)
        warmup = max(1, int(total * self.warmup_fraction))
        floor = self.min_lr / self.lr

        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            phase = min(1, (step - warmup) / max(1, total - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * phase))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def sample(
        self,
        z,
        bond_index,
        batch,
        node_attr=None,
        edge_attr=None,
        n_timesteps=50,
        sampler_type="ode",
        terminal_time=0.99,
        diffusion_scale=0.3,
        probes=2,
        seed=42,
        return_trajectory=False,
    ):
        num_mols = int(batch.max()) + 1
        gen = torch.Generator(device=z.device).manual_seed(seed)
        x = torch.randn((len(z), 3), device=z.device, generator=gen) * self.source_scale

        def predict(state, time):
            t = time.expand(num_mols, 1)
            v, lv = self.predict_posterior(z, t, state, bond_index, batch, node_attr, edge_attr)
            return v, lv.exp()

        # Inference probability and input derivatives are FP32, independently of training autocast.
        with torch.autocast(device_type=z.device.type, enabled=False):
            result, frames = integrate(
                predict,
                x,
                batch,
                steps=n_timesteps,
                terminal_time=terminal_time,
                mode=sampler_type,
                diffusion_scale=diffusion_scale,
                probes=probes,
                source_scale=self.source_scale,
                seed=seed + 100,
                save_trajectory=return_trajectory,
            )
        return (result, frames) if return_trajectory else result
