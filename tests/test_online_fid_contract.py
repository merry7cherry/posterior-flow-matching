from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch_fidelity.metric_fid import fid_features_to_statistics
from torch_fidelity.metric_isc import isc_features_to_metric

from pfm.evaluation import fid_metrics


def test_online_fid_statistics_match_pinned_backend_feature_statistics() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(37, 11, generator=generator, dtype=torch.float32)
    online = fid_metrics._OnlineFeatureStatistics()
    for chunk in features.split((3, 7, 5, 11, 11)):
        online.update(chunk)

    expected = fid_features_to_statistics(features)
    actual = online.finalize()
    assert np.allclose(actual["mu"], expected["mu"], rtol=1e-7, atol=5e-8)
    assert np.allclose(actual["sigma"], expected["sigma"], rtol=1e-10, atol=1e-10)


def test_online_inception_score_matches_pinned_backend_split_semantics() -> None:
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(50, 13, generator=generator, dtype=torch.float32)
    online = fid_metrics._OnlineInceptionScore(
        total_samples=50,
        splits=10,
        rng_seed=2020,
    )
    start = 0
    for chunk in logits.split((7, 9, 3, 12, 19)):
        online.update(chunk, start_index=start)
        start += int(chunk.shape[0])

    expected = isc_features_to_metric(logits, splits=10, shuffle=True, rng_seed=2020)
    actual = online.finalize()
    assert actual["mean"] == pytest.approx(expected["inception_score_mean"], rel=1e-12, abs=1e-12)
    assert actual["std"] == pytest.approx(expected["inception_score_std"], rel=1e-12, abs=1e-12)


def test_generated_common_fid_path_has_no_png_spool_or_feature_list() -> None:
    source = inspect.getsource(fid_metrics.LTH14OnlineMetrics)
    assert "TemporaryDirectory" not in source
    assert "Image.save" not in source
    assert "torch.cat(features" not in source


def test_online_metrics_update_uses_the_canonical_quantizer(monkeypatch) -> None:
    metrics = object.__new__(fid_metrics.LTH14OnlineMetrics)
    metrics.image_size = 32
    metrics.total_samples = 2
    metrics.batch_size = 2
    metrics._buffer = []
    metrics._buffer_count = 0
    metrics._seen = 0
    images = torch.zeros(1, 3, 32, 32)
    calls = []

    def quantize(values, *, image_size):
        calls.append((values, image_size))
        return torch.zeros(1, 3, 32, 32, dtype=torch.uint8)

    monkeypatch.setattr(
        fid_metrics,
        "_quantize_generated_images_for_lth14",
        quantize,
    )

    metrics.update(images)

    assert len(calls) == 1
    assert calls[0][0] is images
    assert calls[0][1] == 32


def test_full_weights_mismatch_fails_before_feature_extractor_work(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        fid_metrics,
        "torch_fidelity_inception_weights_identity",
        lambda: {"path": str(tmp_path / "weights"), "sha256": "a" * 64},
    )
    monkeypatch.setattr(
        fid_metrics,
        "require_jit_torch_fidelity",
        lambda: (_ for _ in ()).throw(AssertionError("feature backend initialized")),
    )
    with pytest.raises(ValueError, match="full weights SHA-256"):
        fid_metrics.LTH14OnlineMetrics(
            fid_stats_path=tmp_path / "stats.npz",
            total_samples=50_000,
            batch_size=64,
            image_size=32,
            include_inception_score=True,
            expected_weights_sha256="b" * 64,
            device=torch.device("cpu"),
        )
