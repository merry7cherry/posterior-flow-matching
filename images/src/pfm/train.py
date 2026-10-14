"""PFM training: interpolate, predict, minimize Gaussian NLL, update EMA."""
import argparse
import copy
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from pfm.checkpoint import save_checkpoint
from pfm.config import ImageConfig, load_config, save_config
from pfm.data import make_dataset, set_training_dataset_epoch
from pfm.models.factory import build_model
from pfm.probability import pfm_loss
from pfm.sampling.time_sampling import sample_train_timesteps


@torch.no_grad()
def update_ema(ema, model, decay):
    torch._foreach_mul_(list(ema.parameters()), decay)
    torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - decay)
    torch._foreach_copy_(list(ema.buffers()), list(model.buffers())) if list(ema.buffers()) else None


def train(config, data_root, output, *, device="cpu", resume=None, dataset=None, quiet=False):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if device == "cuda" else device)
    if device.type == "cpu" and config.mixed_precision != "none":
        raise ValueError("CPU training uses mixed_precision=none")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        torch.manual_seed(config.seed + rank)
        model = build_model(config, device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        effective_lr = config.lr
        if config.lr_reference_batch_size is not None:
            effective_lr *= config.batch_size * world / config.lr_reference_batch_size
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=config.weight_decay,
                                      betas=(config.adamw_beta1, config.adamw_beta2), fused=device.type == "cuda")
        start_epoch, step = 1, 0
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"])
            ema.load_state_dict(saved["ema"])
            optimizer.load_state_dict(saved["optimizer"])
            start_epoch, step = saved["epoch"] + 1, saved["step"]
            rng = saved["rng_states"][rank % len(saved["rng_states"])]
            torch.set_rng_state(rng["cpu"])
            if device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state(rng["cuda"], device)
        if config.compile_model:
            model.compile()
        forward_model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else model
        dataset = make_dataset(config, data_root, train=True) if dataset is None else dataset
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=config.seed, drop_last=config.drop_last) if world > 1 else RandomSampler(dataset)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers,
                            drop_last=config.drop_last, pin_memory=device.type == "cuda", persistent_workers=config.num_workers > 0,
                            generator=torch.Generator().manual_seed(config.seed))
        if len(loader) == 0:
            raise ValueError("dataset is smaller than a training batch")
        output = Path(output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            save_config(config, output / "config.yaml")
        for epoch in range(start_epoch, config.epochs + 1):
            set_training_dataset_epoch(dataset, epoch)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            forward_model.train()
            total_loss, count = 0.0, 0
            for batch_index, (x1, labels) in enumerate(loader):
                progress = epoch - 1 + batch_index / len(loader)
                lr = effective_lr * min(1.0, progress / config.warmup_epochs) if config.warmup_epochs else effective_lr
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                x1, labels = x1.to(device), labels.to(device)
                x0 = config.noise_scale * torch.randn_like(x1)
                t = sample_train_timesteps(config, len(x1), device)
                xt = (1 - t) * x0 + t * x1
                with torch.autocast("cuda", dtype=torch.bfloat16) if config.mixed_precision == "bf16" else nullcontext():
                    loss = pfm_loss(forward_model(xt, t.flatten(), labels), x0, x1, xt, t, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                update_ema(ema, model, config.ema_decay)
                step += 1
                total_loss += loss.detach().item() * len(x1)
                count += len(x1)
                if config.limit_train_batches and batch_index + 1 >= config.limit_train_batches:
                    break
            totals = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(totals)
            rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            rng_states = [None] * world
            if world > 1:
                dist.all_gather_object(rng_states, rng)
            else:
                rng_states[0] = rng
            if rank == 0:
                save_checkpoint(output / "latest.pt", model, ema, optimizer, config, epoch, step, rng_states)
                if epoch % config.checkpoint_every_epochs == 0:
                    save_checkpoint(output / f"epoch_{epoch:04d}.pt", model, ema, optimizer, config, epoch, step, rng_states)
                record = {"epoch": epoch, "step": step, "loss": (totals[0] / totals[1]).item(), "lr": lr}
                with (output / "loss.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if not quiet:
                    print(record, flush=True)
        return output / "latest.pt"
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config, args.set)
    elif args.resume:
        import yaml
        values = torch.load(args.resume, map_location="cpu", weights_only=True)["config"]
        for item in args.set:
            key, value = item.split("=", 1)
            values[key] = yaml.safe_load(value)
        config = ImageConfig(**values)
    else:
        parser.error("provide --config for a new model or --resume for continuation")
    train(config, args.data_root, args.output, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
