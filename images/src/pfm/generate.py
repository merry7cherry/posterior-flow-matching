"""Batched class-conditional generation shared by samples and FID/IS."""
import torch
from pfm.sampling.samplers import initial_noise, sample


def generate_batches(model, config, sampling, device, *, count, batch_size=64, seed=321, capture_time=None):
    for start in range(0, count, batch_size):
        indices = torch.arange(start, min(start + batch_size, count))
        labels = indices % config.num_classes
        yield sample(model, labels, config, sampling, device, seed=seed,
                     initial_noise=initial_noise(config, indices, seed), sample_indices=indices,
                     capture_time=capture_time)
