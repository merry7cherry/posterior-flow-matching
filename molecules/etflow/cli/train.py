"""Train molecular PFM with ordinary Lightning epoch checkpoints."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from etflow.data.datamodule import BaseDataModule
from etflow.models.pfm_model import PosteriorFlow


class TrainingState(pl.Callback):
    """Save the configuration and RNG states alongside Lightning's optimizer state."""
    def __init__(self, config, stop_after_epoch=None):
        self.config = config
        self.stop_after_epoch = stop_after_epoch
        self.pending_rng = None

    def on_train_start(self, trainer, module):
        if self.pending_rng is not None:
            state = self.pending_rng
            torch.set_rng_state(state['torch'].cpu())
            np.random.set_state(state['numpy'])
            random.setstate(state['python'])
            trainer.datamodule.dataloader_args['generator'].set_state(state['loader'].cpu())
            if module.device.type == 'cuda' and 'cuda' in state:
                torch.cuda.set_rng_state(state['cuda'].cpu())
            self.pending_rng = None

    def on_fit_start(self, trainer, module):
        trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch, self.config['seed'])

    def on_train_epoch_start(self, trainer, module):
        trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch, self.config['seed'])

    def on_train_epoch_end(self, trainer, module):
        trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch + 1, self.config['seed'])
        if self.stop_after_epoch is not None and trainer.current_epoch + 1 >= self.stop_after_epoch:
            trainer.should_stop = True

    def on_save_checkpoint(self, trainer, module, checkpoint):
        state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                     python=random.getstate(),
                     loader=trainer.datamodule.dataloader_args['generator'].get_state())
        if module.device.type == 'cuda':
            state['cuda'] = torch.cuda.get_rng_state()
        checkpoint['pfm_rng_state'] = state
        checkpoint['pfm_config'] = self.config

    def on_load_checkpoint(self, trainer, module, checkpoint):
        self.pending_rng = checkpoint['pfm_rng_state']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-epoch', type=int, help='Stop at an epoch boundary; keep the configured schedule')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cpu:
        cfg['trainer'].update(accelerator='cpu', devices=1, precision='32-true', strategy='auto')
    cfg['data_dir'] = str(Path(args.data_dir).resolve())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    pl.seed_everything(cfg['seed'], workers=True)
    dm = BaseDataModule(data_dir=Path(cfg['data_dir']), partition='qm9', dataloader_args={
        **cfg['data'], 'pin_memory': not args.cpu,
        'persistent_workers': cfg['data']['num_workers'] > 0,
        'generator': torch.Generator().manual_seed(cfg['seed']),
    })
    model = PosteriorFlow(**cfg['model_args'])
    checkpoint = ModelCheckpoint(dirpath=str(run_dir/'checkpoints'), filename='epoch-{epoch:03d}',
        auto_insert_metric_name=False, save_last=True, save_top_k=1, monitor='val/loss', mode='min')
    trainer = pl.Trainer(**cfg['trainer'], default_root_dir=str(run_dir),
        logger=CSVLogger(str(run_dir), name='metrics'),
        callbacks=[TrainingState(cfg, args.stop_after_epoch), checkpoint,
                   LearningRateMonitor(logging_interval='step')], enable_progress_bar=False)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    trainer.save_checkpoint(str(run_dir/'checkpoints/final.ckpt'))


if __name__ == '__main__':
    main()
