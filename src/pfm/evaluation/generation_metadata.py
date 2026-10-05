from __future__ import annotations

from pfm.sampling.randomness import RANDOM_PROTOCOL


INITIAL_NOISE_BLOCK_SIZE = 64


def initial_noise_policy(seed: int, *, block_size: int = INITIAL_NOISE_BLOCK_SIZE) -> dict[str, object]:
    return {
        "name": "global_sample_index_seeded_blocks",
        "seed": int(seed),
        "block_size": int(block_size),
        "batch_size_invariant": True,
    }


def sampling_randomness_policy(seed: int) -> dict[str, object]:
    return {
        "version": 1,
        "step_stream_protocol": RANDOM_PROTOCOL,
        "seed": int(seed),
        "initialization": initial_noise_policy(seed),
        "brownian": "global_sample_index_step_seeded",
        "derivative_probes": "global_sample_index_step_probe_seeded",
        "independent_streams": True,
        "batch_partition_invariant_random_inputs": True,
        "cross_device_bitwise_outputs_guaranteed": False,
    }
