"""Read the native Zarr demonstrations into memory for sequence sampling."""
from pathlib import Path
import zarr


class ReplayBuffer:
    def __init__(self, data, episode_ends):
        self.data = data
        self.episode_ends = episode_ends

    @classmethod
    def copy_from_path(cls, zarr_path, keys):
        group = zarr.open(str(Path(zarr_path).expanduser()), mode='r')
        return cls({key: group['data'][key][:] for key in keys},
                   group['meta']['episode_ends'][:])

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def n_steps(self):
        return int(self.episode_ends[-1])

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]
