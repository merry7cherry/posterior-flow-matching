from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_COLORS = {"velocity": "#0F4D92", "endpoint": "#E66100"}


def write_posterior_figures(
    artifact_root: str | Path,
    controlled: Mapping[str, object],
    random_time: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    root = Path(artifact_root)
    figures = root / "posterior_figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {
        "controlled_variance_error": _controlled_variance_error(
            figures, controlled
        ),
        "controlled_complete_nll": _controlled_nll(figures, controlled),
        "random_variance_calibration": _random_calibration(figures, random_time),
        "random_error_variance_ratio": _random_ratio(figures, random_time),
        "random_variance_histogram": _random_histogram(figures, random_time),
        "random_time_calibration": _random_time(figures, random_time),
    }
    return {
        name: {
            "png": path["png"].relative_to(root).as_posix(),
            "pdf": path["pdf"].relative_to(root).as_posix(),
        }
        for name, path in outputs.items()
    }


def _controlled_rows(controlled: Mapping[str, object]) -> list[Mapping[str, object]]:
    by_t = controlled.get("by_t")
    if not isinstance(by_t, Mapping) or not by_t:
        raise ValueError("controlled posterior result must contain non-empty by_t")
    rows = [value for value in by_t.values() if isinstance(value, Mapping)]
    if len(rows) != len(by_t):
        raise ValueError("controlled posterior by_t entries must be mappings")
    return sorted(rows, key=lambda row: float(row["t"]))


def _controlled_variance_error(
    figures: Path,
    controlled: Mapping[str, object],
) -> dict[str, Path]:
    rows = _controlled_rows(controlled)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, space in zip(axes, ("velocity", "endpoint")):
        summaries = [_mapping(row.get(space), f"controlled {space}") for row in rows]
        times = [float(row["t"]) for row in rows]
        axis.plot(times, [_finite(item["mean_pred_var"]) for item in summaries], label="predicted variance")
        axis.plot(times, [_finite(item["mean_squared_error"]) for item in summaries], label="squared error")
        axis.set_title(space.title())
        axis.set_xlabel("t (noise to data)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("component average")
    axes[0].legend()
    return _save(fig, figures / "controlled_variance_error")


def _controlled_nll(
    figures: Path,
    controlled: Mapping[str, object],
) -> dict[str, Path]:
    rows = _controlled_rows(controlled)
    fig, axis = plt.subplots(figsize=(6, 4))
    times = [float(row["t"]) for row in rows]
    for space in ("velocity", "endpoint"):
        values = [
            _finite(_mapping(row.get("variance_nll"), "controlled NLL")[f"{space}_complete_nll_per_dim"])
            for row in rows
        ]
        axis.plot(times, values, label=space, color=_COLORS[space])
    axis.set_xlabel("t (noise to data)")
    axis.set_ylabel("complete NLL / dimension")
    axis.grid(alpha=0.2)
    axis.legend()
    return _save(fig, figures / "controlled_complete_nll")


def _random_spaces(random_time: Mapping[str, object], key: str) -> Mapping[str, object]:
    payload = _mapping(random_time.get(key), f"random-time {key}")
    if set(payload) != {"velocity", "endpoint"}:
        raise ValueError(f"random-time {key} must contain velocity and endpoint")
    return payload


def _random_calibration(figures: Path, random_time: Mapping[str, object]) -> dict[str, Path]:
    spaces = _random_spaces(random_time, "variance_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, space in zip(axes, ("velocity", "endpoint")):
        bins = _bins(spaces[space], "bins")
        predicted = [_finite(row["mean_pred_var"]) for row in bins]
        observed = [_finite(row["mean_squared_error"]) for row in bins]
        axis.plot(predicted, observed, marker="o", color=_COLORS[space])
        bounds = predicted + observed
        low, high = min(bounds), max(bounds)
        axis.plot([low, high], [low, high], linestyle="--", color="#555555")
        axis.set_title(space.title())
        axis.set_xlabel("predicted variance")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("observed squared error")
    return _save(fig, figures / "random_variance_calibration")


def _random_ratio(figures: Path, random_time: Mapping[str, object]) -> dict[str, Path]:
    spaces = _random_spaces(random_time, "variance_calibration")
    fig, axis = plt.subplots(figsize=(6, 4))
    for space in ("velocity", "endpoint"):
        bins = _bins(spaces[space], "bins")
        axis.plot(
            [_finite(row["mean_pred_var"]) for row in bins],
            [_finite(row["error_to_var_ratio"]) for row in bins],
            marker="o",
            label=space,
            color=_COLORS[space],
        )
    axis.axhline(1.0, linestyle="--", color="#555555")
    axis.set_xscale("log")
    axis.set_xlabel("predicted variance")
    axis.set_ylabel("error / variance")
    axis.grid(alpha=0.2)
    axis.legend()
    return _save(fig, figures / "random_error_variance_ratio")


def _random_histogram(figures: Path, random_time: Mapping[str, object]) -> dict[str, Path]:
    spaces = _random_spaces(random_time, "variance_calibration")
    fig, axis = plt.subplots(figsize=(6, 4))
    for space in ("velocity", "endpoint"):
        bins = _bins(spaces[space], "histogram_bins")
        axis.plot(
            [_finite(row["center_pred_var"]) for row in bins],
            [_finite(row["fraction"]) for row in bins],
            marker="o",
            label=space,
            color=_COLORS[space],
        )
    axis.set_xscale("log")
    axis.set_xlabel("predicted variance")
    axis.set_ylabel("sample fraction")
    axis.grid(alpha=0.2)
    axis.legend()
    return _save(fig, figures / "random_variance_histogram")


def _random_time(figures: Path, random_time: Mapping[str, object]) -> dict[str, Path]:
    spaces = _random_spaces(random_time, "time_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, space in zip(axes, ("velocity", "endpoint")):
        bins = _bins(spaces[space], "bins")
        times = [_finite(row["t_mean"]) for row in bins]
        axis.plot(times, [_finite(row["mean_pred_var"]) for row in bins], label="predicted variance")
        axis.plot(times, [_finite(row["mean_squared_error"]) for row in bins], label="squared error")
        axis.set_title(space.title())
        axis.set_xlabel("random t (noise to data)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("component average")
    axes[0].legend()
    return _save(fig, figures / "random_time_calibration")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _bins(value: object, key: str) -> list[Mapping[str, object]]:
    payload = _mapping(value, "posterior space")
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"posterior space {key} must be a non-empty list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"posterior space {key} entries must be mappings")
    return rows


def _finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("posterior figure values must be finite")
    return result


def _save(fig, stem: Path) -> dict[str, Path]:
    fig.tight_layout()
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}
