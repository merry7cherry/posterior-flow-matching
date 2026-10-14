import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import numpy as np
import pytest
import torch
import zarr

torch.set_num_threads(2)

@pytest.fixture
def native_data(tmp_path):
    """Synthetic native-format storage fixture; not a task-quality benchmark."""
    if os.environ.get('PFM_TEST_DATA'):
        return os.environ['PFM_TEST_DATA']
    path=tmp_path/'fixture.zarr'
    group=zarr.open(str(path),mode='w')
    rng=np.random.default_rng(1)
    group.create_dataset('data/img',data=rng.integers(0,256,(80,96,96,3),dtype=np.uint8))
    group.create_dataset('data/state',data=rng.uniform(100,400,(80,5)).astype('float32'))
    group.create_dataset('data/action',data=rng.uniform(100,400,(80,2)).astype('float32'))
    group.create_dataset('meta/episode_ends',data=np.arange(16,81,16))
    return path
