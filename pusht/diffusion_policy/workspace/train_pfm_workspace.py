import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
from diffusion_policy.policy.pfm_image_policy import PFMImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler


class PFMWorkspace:

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: PFMImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: PFMImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.resume_state = None

    @property
    def output_dir(self):
        if self._output_dir is not None:
            return self._output_dir
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir) / 'checkpoints' / f'{tag}.ckpt'

    def save_checkpoint(self, tag='latest'):
        target = self.get_checkpoint_path(tag)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict() if self.ema_model is not None else None,
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step, 'epoch': self.epoch,
            'resume_state': self.resume_state}
        temporary = target.with_suffix('.ckpt.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return str(target)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(payload['model'])
        if self.ema_model is not None:
            self.ema_model.load_state_dict(payload['ema_model'])
        self.optimizer.load_state_dict(payload['optimizer'])
        self.global_step, self.epoch = payload['global_step'], payload['epoch']
        self.resume_state = payload['resume_state']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            self.load_checkpoint(cfg.training.resume)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.resume_state:
            self.epoch = self.resume_state['next_epoch']
            self.global_step = self.resume_state['next_step']

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                min(len(train_dataloader), cfg.training.max_train_steps or len(train_dataloader)) * cfg.training.num_epochs),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        if self.resume_state:
            lr_scheduler.load_state_dict(self.resume_state['lr_scheduler'])
            for group, lr in zip(self.optimizer.param_groups, lr_scheduler.get_last_lr()):
                group['lr'] = lr
            if ema is not None:
                ema.optimization_step = self.resume_state['ema_step']

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            if self.resume_state:
                random.setstate(self.resume_state['python_rng'])
                np.random.set_state(self.resume_state['numpy_rng'])
                torch.set_rng_state(self.resume_state['torch_rng'].cpu())
                if device.type == 'cuda':
                    torch.cuda.set_rng_state_all([x.cpu() for x in self.resume_state['cuda_rng']])
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite PFM training loss')
                        loss.backward()
                        gradient_stats = self.model.clip_gradients(cfg.training.max_grad_norm)

                        # step optimizer
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        step_log.update({k: float(v) for k, v in self.model.loss_stats.items()})
                        step_log.update({k: float(v) for k, v in gradient_stats.items()})

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if cfg.training.max_train_steps is not None:
                            is_last_batch |= batch_idx >= cfg.training.max_train_steps - 1
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run PFM sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                self.resume_state = {
                    'next_epoch': self.epoch + 1, 'next_step': self.global_step + 1,
                    'ema_step': ema.optimization_step if ema is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint()
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
            self.save_checkpoint(tag='final')
            self.save_checkpoint()
            env_runner.env.close()
