"""Streaming FID and IS using the pinned LTH14 Inception implementation."""
import math
import numpy as np
import torch


def feature_extractor(device, *, inception_score=False, weights=None):
    from torch_fidelity.defaults import DEFAULTS
    from torch_fidelity.utils import create_feature_extractor, resolve_feature_extractor, resolve_feature_layer_for_metric
    options = dict(DEFAULTS)
    options.update(cuda=torch.device(device).type == "cuda", fid=True, isc=inception_score,
                   verbose=False, feature_extractor_weights_path=weights)
    fid_layer = resolve_feature_layer_for_metric("fid", **options)
    isc_layer = resolve_feature_layer_for_metric("isc", **options) if inception_score else None
    layers = [fid_layer] + ([isc_layer] if isc_layer else [])
    return create_feature_extractor(resolve_feature_extractor(**options), layers, **options), fid_layer, isc_layer


def quantize(images):
    return torch.round((images.clamp(-1, 1) + 1) * 127.5).clamp(0, 255).to(torch.uint8)


class GenerationMetrics:
    def __init__(self, stats_path, num_samples, device, *, inception_score=False, weights=None):
        self.extractor, self.fid_layer, self.isc_layer = feature_extractor(device, inception_score=inception_score, weights=weights)
        with np.load(stats_path) as stats:
            self.reference = {"mu": stats["mu"], "sigma": stats["sigma"]}
        self.device, self.seen = device, 0
        self.features = _OnlineFeatureStatistics()
        self.isc = _OnlineInceptionScore(total_samples=num_samples, splits=10, rng_seed=2020) if inception_score else None

    @torch.no_grad()
    def update(self, images):
        values = self.extractor.convert_features_tuple_to_dict(self.extractor(quantize(images).to(self.device)))
        self.features.update(values[self.fid_layer])
        if self.isc is not None:
            self.isc.update(values[self.isc_layer], start_index=self.seen)
        self.seen += len(images)

    def finalize(self):
        from torch_fidelity.metric_fid import fid_statistics_to_metric, KEY_METRIC_FID
        result = {"fid": float(fid_statistics_to_metric(self.reference, self.features.finalize(), verbose=False)[KEY_METRIC_FID])}
        if self.isc is not None:
            result["inception_score"] = self.isc.finalize()
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
