import copy
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv


def make_policy():
    cfg=OmegaConf.load('diffusion_policy/config/pusht.yaml')
    cfg.policy.down_dims=[16,32]
    cfg.policy.diffusion_step_embed_dim=16
    cfg.policy.sampling.steps=2
    policy=hydra.utils.instantiate(cfg.policy)
    normalizer=LinearNormalizer()
    for key in ('action','agent_pos'):
        normalizer[key]=SingleFieldLinearNormalizer.create_identity()
    normalizer['image']=SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def test_native_env_and_pfm_output_slice():
    env=PushTImageEnv()
    env.seed(100000)
    obs=env.reset()
    assert obs['image'].shape==(3,96,96)
    env.step(np.array([256.,256.]))
    env.close()
    policy=make_policy().eval()
    obs={k:torch.from_numpy(np.stack([v,v]))[None].float() for k,v in obs.items()}
    output=policy.predict_action(obs)
    assert output['action_pred'].shape==(1,16,2)
    torch.testing.assert_close(output['action'],output['action_pred'][:,1:9])
    assert policy.obs_feature_dim==66
    policy.sampling.update(solver='sde',diffusion_scale=.1,derivative_probes=2)
    with torch.no_grad():sde=policy.predict_action(obs)
    assert torch.isfinite(sde['action']).all()
    assert all(p.grad is None for p in policy.parameters())


def test_training_updates_encoder_and_both_heads():
    policy=make_policy().train()
    batch={'obs':{'image':torch.rand(2,2,3,96,96),'agent_pos':torch.randn(2,2,2)},
           'action':torch.randn(2,16,2)}
    policy.compute_loss(batch).backward()
    for module in (policy.obs_encoder,policy.model.final_conv,policy.model.variance_head):
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in module.parameters())
