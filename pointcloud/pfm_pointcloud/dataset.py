"""Shape-local normalization and deterministic ordering from the native loader."""
from copy import copy
import random
import h5py
import torch
from torch.utils.data import Dataset

SYNSETS = {'chair': '03001627', 'airplane': '02691156'}


class ShapeNetCore(Dataset):
    def __init__(self, path, cates, split, scale_mode='shape_unit', limit=None):
        if not isinstance(cates, list) or len(cates) != 1 or cates[0] not in SYNSETS:
            raise ValueError('select one of chair or airplane')
        if split not in ('train', 'test') or scale_mode not in ('shape_unit', 'shape_bbox'):
            raise ValueError('unsupported split or normalization')
        if limit is not None and limit < 1:
            raise ValueError('limit must be positive')
        self.pointclouds = []
        category = cates[0]
        with h5py.File(path, 'r') as handle:
            data = handle[SYNSETS[category]][split]
            indices = list(range(len(data)))
            random.Random(2020).shuffle(indices)
            for index in indices[:limit]:
                pc = torch.from_numpy(data[index]).float()
                if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) < 2 or not torch.isfinite(pc).all():
                    raise ValueError('expected finite [N,3] point clouds')
                if scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                else:
                    low, high = pc.min(0, keepdim=True).values, pc.max(0, keepdim=True).values
                    shift = ((low + high) / 2).view(1, 3)
                    scale = (high - low).max().reshape(1, 1) / 2
                if (scale <= 0).any():
                    raise ValueError('degenerate point cloud')
                self.pointclouds.append({'pointcloud': (pc-shift)/scale, 'cate': category,
                                         'id': index, 'shift': shift, 'scale': scale})
        if not self.pointclouds:
            raise ValueError('dataset split is empty')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, index):
        return {k: v.clone() if isinstance(v, torch.Tensor) else copy(v)
                for k, v in self.pointclouds[index].items()}
