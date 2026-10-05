from __future__ import annotations

import hashlib
import hmac
import importlib
import types
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


_TORCH_FIDELITY_INCEPTION_WEIGHTS_NAME = "weights-inception-2015-12-05-6726825d.pth"
_TORCH_FIDELITY_INCEPTION_SHA256_PREFIX = "6726825d"


class LTH14OnlineMetrics:
    """Feed generated uint8 batches through the pinned LTH14 extractor online."""

    def __init__(
        self,
        *,
        fid_stats_path: str | Path,
        total_samples: int,
        batch_size: int,
        image_size: int,
        include_inception_score: bool,
        expected_weights_sha256: str,
        device: torch.device,
    ) -> None:
        if isinstance(total_samples, bool) or not isinstance(total_samples, int) or total_samples < 2:
            raise ValueError("LTH14 online metrics require at least two samples")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("LTH14 online batch_size must be a positive integer")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
            raise ValueError("LTH14 online image_size must be a positive integer")
        weights = torch_fidelity_inception_weights_identity()
        if not hmac.compare_digest(weights["sha256"], expected_weights_sha256):
            raise ValueError("LTH14 Inception full weights SHA-256 does not match bound FID asset")
        require_jit_torch_fidelity()
        from torch_fidelity.defaults import DEFAULTS
        from torch_fidelity.utils import (
            create_feature_extractor,
            resolve_feature_extractor,
            resolve_feature_layer_for_metric,
        )

        kwargs = dict(DEFAULTS)
        kwargs.update(
            {
                "cuda": device.type == "cuda",
                "batch_size": batch_size,
                "isc": bool(include_inception_score),
                "fid": True,
                "kid": False,
                "prc": False,
                "ppl": False,
                "feature_extractor_weights_path": weights["path"],
                "verbose": False,
            }
        )
        extractor_name = resolve_feature_extractor(**kwargs)
        self.fid_layer = resolve_feature_layer_for_metric("fid", **kwargs)
        layers = [self.fid_layer]
        self.isc_layer: str | None = None
        if include_inception_score:
            self.isc_layer = resolve_feature_layer_for_metric("isc", **kwargs)
            layers.append(self.isc_layer)
        self.extractor = create_feature_extractor(extractor_name, layers, **kwargs)
        real = load_fid_statistics(fid_stats_path)
        self.real_statistics = {
            "mu": real["mean"].detach().cpu().numpy(),
            "sigma": real["cov"].detach().cpu().numpy(),
        }
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self._buffer: list[torch.Tensor] = []
        self._buffer_count = 0
        self._seen = 0
        self._fid = _OnlineFeatureStatistics()
        self._isc = (
            _OnlineInceptionScore(total_samples=total_samples, splits=10, rng_seed=2020)
            if include_inception_score
            else None
        )

    @torch.no_grad()
    def update(self, images: torch.Tensor) -> None:
        quantized = _quantize_generated_images_for_lth14(
            images,
            image_size=self.image_size,
        )
        if self._seen + self._buffer_count + int(quantized.shape[0]) > self.total_samples:
            raise ValueError("LTH14 online stream exceeds declared population")
        self._buffer.append(quantized)
        self._buffer_count += int(quantized.shape[0])
        while self._buffer_count >= self.batch_size:
            self._flush(self.batch_size)

    @torch.no_grad()
    def _flush(self, count: int) -> None:
        joined = torch.cat(self._buffer, dim=0)
        batch = joined[:count]
        remainder = joined[count:]
        self._buffer = [remainder] if int(remainder.shape[0]) else []
        self._buffer_count = int(remainder.shape[0])
        features_tuple = self.extractor(batch.to(self.device, non_blocking=self.device.type == "cuda"))
        features = self.extractor.convert_features_tuple_to_dict(features_tuple)
        fid_features = features[self.fid_layer].detach().cpu()
        self._fid.update(fid_features)
        if self._isc is not None:
            assert self.isc_layer is not None
            self._isc.update(features[self.isc_layer].detach().cpu(), start_index=self._seen)
        self._seen += count

    def finalize(self) -> dict[str, object]:
        if self._buffer_count:
            self._flush(self._buffer_count)
        if self._seen != self.total_samples:
            raise ValueError(
                f"LTH14 online stream produced {self._seen} samples, expected {self.total_samples}"
            )
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        from torch_fidelity.metric_fid import KEY_METRIC_FID

        generated = self._fid.finalize()
        fid = fid_statistics_to_metric(self.real_statistics, generated, verbose=False)
        result: dict[str, object] = {"fid": float(fid[KEY_METRIC_FID])}
        if self._isc is not None:
            result["inception_score"] = self._isc.finalize()
        return result


class _OnlineFeatureStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, features: torch.Tensor) -> None:
        matrix = features.detach().cpu().numpy().astype(np.float64, copy=False)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
            raise ValueError("online FID features must be a finite matrix")
        batch_count = int(matrix.shape[0])
        batch_mean = matrix.mean(axis=0, dtype=np.float64)
        centered = matrix - batch_mean
        batch_m2 = centered.T @ centered
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.m2 is not None
        if matrix.shape[1] != self.mean.shape[0]:
            raise ValueError("online FID feature dimension changed")
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + np.outer(delta, delta) * (
            self.count * batch_count / combined_count
        )
        self.mean += delta * (batch_count / combined_count)
        self.count = combined_count

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count < 2 or self.mean is None or self.m2 is None:
            raise ValueError("online FID requires at least two feature rows")
        mu = self.mean
        sigma = self.m2 / (self.count - 1)
        sigma = 0.5 * (sigma + sigma.T)
        return {"mu": mu, "sigma": sigma}


class _OnlineInceptionScore:
    def __init__(self, *, total_samples: int, splits: int, rng_seed: int) -> None:
        if total_samples < splits:
            raise ValueError("online Inception Score population must cover every split")
        permutation = np.random.RandomState(rng_seed).permutation(total_samples)
        assignment = np.empty(total_samples, dtype=np.int64)
        for split in range(splits):
            assignment[permutation[split * total_samples // splits : (split + 1) * total_samples // splits]] = split
        self.assignment = assignment
        self.splits = splits
        self.counts = np.zeros(splits, dtype=np.int64)
        self.p_sums: list[torch.Tensor | None] = [None] * splits
        self.p_log_p_sums = np.zeros(splits, dtype=np.float64)

    def update(self, logits: torch.Tensor, *, start_index: int) -> None:
        values = logits.detach().cpu().double()
        if values.ndim != 2 or not bool(torch.isfinite(values).all()):
            raise ValueError("online Inception Score logits must be a finite matrix")
        stop = start_index + int(values.shape[0])
        if stop > int(self.assignment.shape[0]):
            raise ValueError("online Inception Score stream exceeds declared population")
        probabilities = values.softmax(dim=1)
        log_probabilities = values.log_softmax(dim=1)
        assignments = self.assignment[start_index:stop]
        for split in range(self.splits):
            positions = np.flatnonzero(assignments == split)
            if positions.size == 0:
                continue
            selected = probabilities[torch.as_tensor(positions, dtype=torch.long)]
            selected_log = log_probabilities[torch.as_tensor(positions, dtype=torch.long)]
            contribution = selected.sum(dim=0)
            self.p_sums[split] = (
                contribution
                if self.p_sums[split] is None
                else self.p_sums[split] + contribution
            )
            self.p_log_p_sums[split] += float((selected * selected_log).sum())
            self.counts[split] += int(positions.size)

    def finalize(self) -> dict[str, float]:
        scores = []
        for split in range(self.splits):
            count = int(self.counts[split])
            p_sum = self.p_sums[split]
            if count < 1 or p_sum is None:
                raise ValueError("online Inception Score split is empty")
            q = p_sum / count
            mean_kl = (
                self.p_log_p_sums[split] - float((p_sum * q.log()).sum())
            ) / count
            scores.append(math.exp(mean_kl))
        return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def _quantize_generated_images_for_lth14(
    images: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Apply the canonical generated-image quantization without resizing."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("LTH14 FID images must be a [N,C,H,W] tensor")
    if images.shape[1:] != (3, image_size, image_size):
        raise ValueError("LTH14 FID images must retain their bound RGB geometry")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("LTH14 FID images must contain only finite values")
    values = images.detach().cpu()
    return torch.round((values.clamp(-1.0, 1.0) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)


def validate_torch_fidelity_inception_weights() -> Path:
    return _validate_local_inception_weights(
        _TORCH_FIDELITY_INCEPTION_WEIGHTS_NAME,
        _TORCH_FIDELITY_INCEPTION_SHA256_PREFIX,
        "LTH14 Inception",
    )


def prepare_inception_weights() -> dict[str, str]:
    """Explicitly download the pinned extractor weights into the active hub cache."""
    require_jit_torch_fidelity()
    from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3

    hub_dir = Path(torch.hub.get_dir())
    checkpoint_dir = hub_dir / "checkpoints"
    if hub_dir.is_symlink() or checkpoint_dir.is_symlink():
        raise ValueError("LTH14 Inception weight cache directory must not be a symlink")
    path = checkpoint_dir / _TORCH_FIDELITY_INCEPTION_WEIGHTS_NAME
    if path.exists() or path.is_symlink():
        return torch_fidelity_inception_weights_identity()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".inception-", dir=checkpoint_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.hub.download_url_to_file(
            URL_INCEPTION_V3,
            str(temporary),
            hash_prefix=_TORCH_FIDELITY_INCEPTION_SHA256_PREFIX,
            progress=True,
        )
        # Publish complete bytes without replacing a concurrent writer's file.
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        return torch_fidelity_inception_weights_identity()
    finally:
        temporary.unlink(missing_ok=True)


def torch_fidelity_inception_weights_identity() -> dict[str, str]:
    path = validate_torch_fidelity_inception_weights()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "filename": path.name, "sha256": digest}


def _validate_local_inception_weights(
    filename: str,
    expected_sha256_prefix: str,
    description: str,
) -> Path:
    hub_dir = Path(torch.hub.get_dir())
    checkpoint_dir = hub_dir / "checkpoints"
    if hub_dir.is_symlink() or checkpoint_dir.is_symlink():
        raise ValueError(f"{description} weight cache directory must not be a symlink")
    path = checkpoint_dir / filename
    if path.is_symlink():
        raise ValueError(f"{description} weight must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} weight does not exist: {path}; "
            "prepare it explicitly with pfm-prepare-fid --download-weights"
        )
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if not digest.startswith(expected_sha256_prefix):
        raise ValueError(
            f"{description} weight digest mismatch: expected SHA-256 prefix "
            f"{expected_sha256_prefix}, got {digest}"
        )
    return path


def require_jit_torch_fidelity() -> types.ModuleType:
    try:
        import torch_fidelity
    except ImportError as exc:
        raise ImportError("LTH14 torch-fidelity is required for JiT ImageNet FID") from exc
    try:
        defaults = importlib.import_module("torch_fidelity.defaults").DEFAULTS
    except (AttributeError, ImportError) as exc:
        raise ImportError("Installed torch-fidelity does not expose DEFAULTS; use the LTH14 fork") from exc
    if "fid_statistics_file" not in defaults:
        raise ImportError(
            "Installed torch-fidelity does not support fid_statistics_file; "
            "install git+https://github.com/LTH14/torch-fidelity.git for JiT ImageNet FID"
        )
    return torch_fidelity


def load_fid_statistics(path: str | Path) -> dict[str, object]:
    stats_path = Path(path)
    if stats_path.is_symlink():
        raise ValueError(f"FID statistics must not be a symlink: {stats_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"FID statistics do not exist: {stats_path}")
    if stats_path.suffix != ".npz":
        raise ValueError("FID statistics must be a canonical .npz file")
    try:
        with np.load(stats_path, allow_pickle=False) as data:
            if set(data.files) != {"mu", "sigma"}:
                raise ValueError("FID statistics NPZ members must be exactly ['mu', 'sigma']")
            mu = np.asarray(data["mu"])
            sigma = np.asarray(data["sigma"])
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("FID statistics NPZ members"):
            raise
        raise ValueError(f"invalid FID statistics NPZ: {stats_path}") from exc
    if mu.ndim != 1 or mu.shape[0] < 1:
        raise ValueError("FID statistics mu must be a non-empty vector")
    if sigma.ndim != 2 or sigma.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError("FID statistics sigma must be a square matrix matching mu")
    if not np.issubdtype(mu.dtype, np.number) or not np.issubdtype(sigma.dtype, np.number):
        raise ValueError("FID statistics mu and sigma must be numeric")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("FID statistics mu and sigma must be finite")
    if not np.allclose(sigma, sigma.T, rtol=1e-7, atol=1e-9):
        raise ValueError("FID statistics sigma must be symmetric")
    return {
        "mean": torch.as_tensor(mu.copy(), dtype=torch.float64),
        "cov": torch.as_tensor(sigma.copy(), dtype=torch.float64),
    }
