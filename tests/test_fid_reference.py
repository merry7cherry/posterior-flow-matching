from __future__ import annotations

import numpy as np

from pfm.evaluation.fid_reference import save_stats


def test_save_stats_publishes_npz_without_temporary_residue(tmp_path):
    output = tmp_path / "fid_stats.npz"

    save_stats(
        output,
        {
            "mu": np.array([1.0], dtype=np.float32),
            "sigma": np.array([[2.0]], dtype=np.float32),
        },
    )

    with np.load(output, allow_pickle=False) as stats:
        np.testing.assert_array_equal(stats["mu"], np.array([1.0], dtype=np.float32))
        np.testing.assert_array_equal(
            stats["sigma"], np.array([[2.0]], dtype=np.float32)
        )
    assert not list(tmp_path.glob(".fid_stats.npz.*"))
