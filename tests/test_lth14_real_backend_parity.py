from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pfm.evaluation.fid_metrics import (
    LTH14OnlineMetrics,
    torch_fidelity_inception_weights_identity,
)


def test_online_tensor_path_matches_real_pinned_backend(tmp_path: Path) -> None:
    try:
        weights = torch_fidelity_inception_weights_identity()
    except FileNotFoundError:
        pytest.skip("Run pfm-prepare-fid --download-weights to obtain the verified LTH14 Inception weights")

    from torch_fidelity.defaults import DEFAULTS
    from torch_fidelity.metric_fid import fid_features_to_statistics
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_features_to_metric
    from torch_fidelity.utils import (
        create_feature_extractor,
        resolve_feature_extractor,
        resolve_feature_layer_for_metric,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(321)
    real = torch.randint(0, 256, (20, 3, 32, 32), dtype=torch.uint8, generator=generator)
    generated = torch.randint(0, 256, (20, 3, 32, 32), dtype=torch.uint8, generator=generator)
    kwargs = dict(DEFAULTS)
    kwargs.update(
        {
            "cuda": device.type == "cuda",
            "batch_size": 5,
            "isc": True,
            "fid": True,
            "kid": False,
            "prc": False,
            "ppl": False,
            "feature_extractor_weights_path": weights["path"],
            "verbose": False,
        }
    )
    name = resolve_feature_extractor(**kwargs)
    layer = resolve_feature_layer_for_metric("fid", **kwargs)
    isc_layer = resolve_feature_layer_for_metric("isc", **kwargs)
    extractor = create_feature_extractor(name, [layer, isc_layer], **kwargs)
    real_features = []
    generated_features = []
    generated_logits = []
    with torch.no_grad():
        for batch in real.split(5):
            extracted = extractor(batch.to(device))
            real_features.append(
                extractor.convert_features_tuple_to_dict(extracted)[layer].cpu()
            )
        for batch in generated.split(5):
            extracted = extractor.convert_features_tuple_to_dict(
                extractor(batch.to(device))
            )
            generated_features.append(extracted[layer].cpu())
            generated_logits.append(extracted[isc_layer].cpu())
    real_stats = fid_features_to_statistics(torch.cat(real_features, dim=0))
    stats_path = tmp_path / "real_stats.npz"
    np.savez(stats_path, mu=real_stats["mu"], sigma=real_stats["sigma"])

    expected_fid = fid_statistics_to_metric(
        real_stats,
        fid_features_to_statistics(torch.cat(generated_features, dim=0)),
        verbose=False,
    )["frechet_inception_distance"]
    expected_isc = isc_features_to_metric(
        torch.cat(generated_logits, dim=0),
        splits=10,
        shuffle=True,
        rng_seed=2020,
    )
    online = LTH14OnlineMetrics(
        fid_stats_path=stats_path,
        total_samples=20,
        batch_size=5,
        image_size=32,
        include_inception_score=True,
        expected_weights_sha256=weights["sha256"],
        device=device,
    )
    generated_float = generated.float() / 127.5 - 1.0
    for chunk in generated_float.split((3, 7, 10)):
        online.update(chunk)
    online_result = online.finalize()

    assert online_result["fid"] == pytest.approx(expected_fid, rel=1e-6, abs=1e-6)
    assert online_result["inception_score"]["mean"] == pytest.approx(
        expected_isc["inception_score_mean"], rel=1e-10, abs=1e-10
    )
    assert online_result["inception_score"]["std"] == pytest.approx(
        expected_isc["inception_score_std"], rel=1e-10, abs=1e-10
    )
