"""Native parallel image observations, physics and video output with pinned Gym."""
import numpy as np
import pytest
import torch
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def make_env():
    return MultiStepWrapper(PushTImageEnv(), n_obs_steps=2,
                            n_action_steps=8, max_episode_steps=8)


@pytest.mark.parametrize('shared_memory', [True, False])
def test_parallel_matches_native_observations_and_physics(shared_memory):
    reference=make_env()
    parallel=AsyncVectorEnv([make_env], shared_memory=shared_memory)
    try:
        reference.seed(42); parallel.seed([42])
        expected=reference.reset(); actual=parallel.reset()
        for key in expected:np.testing.assert_array_equal(actual[key][0],expected[key].astype(reference.observation_space[key].dtype))
        action=np.full((8,2),256.,dtype=np.float32)
        obs,reward,done,_=reference.step(action)
        pobs,preward,pdone,_=parallel.step(action[None])
        for key in obs:np.testing.assert_array_equal(pobs[key][0],obs[key].astype(reference.observation_space[key].dtype))
        assert preward[0]==reward and pdone[0]==done
    finally:
        parallel.close(); reference.close()


class FixedPolicy(BaseImagePolicy):
    def predict_action(self,obs_dict):
        return {'action':torch.full((len(obs_dict['image']),8,2),256.)}


def test_native_runner_produces_scores_and_videos(tmp_path):
    runner=PushTImageRunner(str(tmp_path),n_train=1,n_test=1,n_train_vis=1,
                           n_test_vis=1,n_envs=2,n_obs_steps=2,n_action_steps=8,
                           max_steps=8)
    try:result=runner.run(FixedPolicy())
    finally:runner.env.close()
    assert np.isfinite(result['test/mean_score'])
    videos=list((tmp_path/'media').glob('*.mp4'))
    assert len(videos)==2 and all(p.stat().st_size>0 for p in videos)
