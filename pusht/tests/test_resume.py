"""Real workspace interruption/restoration with small native image network."""
from types import SimpleNamespace
import os
import hydra
import pytest
import torch
from omegaconf import OmegaConf
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.workspace.train_pfm_workspace import PFMWorkspace


class Runner(BaseImageRunner):
    def __init__(self):
        self.env=SimpleNamespace(close=lambda:None)
    def run(self,policy):
        return {'test/mean_score':0.0}


def test_epoch_resume_matches_uninterrupted_weights_and_ema(tmp_path,monkeypatch,native_data):
    # Keep two epochs in the LR horizon, and interrupt after first checkpoint.
    (tmp_path/'whole').mkdir()
    (tmp_path/'resumed').mkdir()
    cfg=OmegaConf.load('diffusion_policy/config/pusht.yaml')
    cfg.task.dataset.zarr_path=str(native_data)
    cfg.policy.logvar_min=-3.
    cfg.policy.logvar_max=3.
    cfg.policy.down_dims=[16,32]
    cfg.policy.diffusion_step_embed_dim=16
    cfg.policy.sampling.steps=1
    cfg.training.device=os.environ.get('PFM_TEST_DEVICE','cpu')
    if os.environ.get('PFM_TEST_DATA'):
        cfg.task.dataset.zarr_path=os.environ['PFM_TEST_DATA']
    if cfg.training.device.startswith('cuda'):
        monkeypatch.setattr(torch.backends.cudnn, 'deterministic', True)
    cfg.training.num_epochs=2
    cfg.training.max_train_steps=1
    cfg.training.max_val_steps=1
    cfg.training.checkpoint_every=1
    cfg.training.rollout_every=1
    cfg.dataloader.batch_size=2
    cfg.dataloader.num_workers=0
    cfg.val_dataloader.batch_size=2
    cfg.val_dataloader.num_workers=0
    cfg.training.resume=None
    original=hydra.utils.instantiate
    def instantiate(config,*args,**kwargs):
        if str(config.get('_target_','')).endswith('PushTImageRunner'):
            return Runner()
        return original(config,*args,**kwargs)
    monkeypatch.setattr(hydra.utils,'instantiate',instantiate)
    whole=PFMWorkspace(cfg,output_dir=str(tmp_path/'whole'))
    whole.run()
    interrupted=PFMWorkspace(cfg,output_dir=str(tmp_path/'resumed'))
    save=interrupted.save_checkpoint
    def stop(*args,**kwargs):
        result=save(*args,**kwargs)
        raise InterruptedError('simulated interruption after completed epoch')
    interrupted.save_checkpoint=stop
    with pytest.raises(InterruptedError):interrupted.run()
    cfg.training.resume=str(tmp_path/'resumed/checkpoints/latest.ckpt')
    resumed=PFMWorkspace(cfg,output_dir=str(tmp_path/'resumed'))
    resumed.run()
    assert whole.epoch==resumed.epoch==2
    assert whole.global_step==resumed.global_step
    assert whole.resume_state['ema_step']==resumed.resume_state['ema_step']
    for name in ('model','ema_model'):
        for key,value in getattr(whole,name).state_dict().items():
            torch.testing.assert_close(value,getattr(resumed,name).state_dict()[key],rtol=0,atol=0,msg=f"{name}.{key}")
