from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pfm.core.atomic_io import atomic_output_path
from pfm.data.datasets import to_display_range


_DPI = 200
_VARIANCE_CMAP = "magma"
_VARIANCE_POSITION_LOG10_RANGE = (-8.0, -2.0)


def save_sample_grid_for_labels(
    samples: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
    ordered_labels: list[int],
    samples_per_label: int,
) -> None:
    images = _image_array(to_display_range(samples))
    labels_array = _labels_array(labels, expected_count=len(images))
    fig, slots = _label_grid(ordered_labels, samples_per_label)
    for class_id, axes in slots.items():
        indices = _class_indices(labels_array, class_id, samples_per_label)
        for axis, index in zip(axes, indices):
            image = images[index]
            if image.ndim == 2:
                axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axis.imshow(np.clip(image, 0.0, 1.0))
    _save_png(fig, path)


def save_raw_variance_grid_for_labels(
    var_maps: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
    ordered_labels: list[int],
    samples_per_label: int,
    *,
    value_range: tuple[float, float],
    colorbar_label: str,
) -> None:
    maps = _map_array(var_maps)
    labels_array = _labels_array(labels, expected_count=len(maps))
    vmin, vmax = map(float, value_range)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        raise ValueError("variance value_range must contain finite increasing values")
    fig, slots = _label_grid(ordered_labels, samples_per_label)
    mappable = None
    for class_id, axes in slots.items():
        indices = _class_indices(labels_array, class_id, samples_per_label)
        for axis, index in zip(axes, indices):
            mappable = axis.imshow(
                maps[index],
                cmap=_VARIANCE_CMAP,
                vmin=vmin,
                vmax=vmax,
            )
    if mappable is not None:
        all_axes = [axis for axes in slots.values() for axis in axes]
        colorbar = fig.colorbar(mappable, ax=all_axes, fraction=0.02, pad=0.01)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    _save_png(fig, path)


def save_variance_position_grid_for_labels(
    var_maps: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
    ordered_labels: list[int],
    samples_per_label: int,
    *,
    value_range: tuple[float, float],
) -> None:
    """Show where variance lies in its authored range with fixed log contrast."""
    maps = _map_array(var_maps)
    labels_array = _labels_array(labels, expected_count=len(maps))
    lower, upper = map(float, value_range)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("variance value_range must contain finite increasing values")
    position = np.clip((maps - lower) / (upper - lower), 0.0, 1.0)
    log10_floor, log10_ceiling = _VARIANCE_POSITION_LOG10_RANGE
    contrast = np.log10(np.maximum(position, 10.0**log10_floor))
    fig, slots = _label_grid(ordered_labels, samples_per_label)
    mappable = None
    for class_id, axes in slots.items():
        indices = _class_indices(labels_array, class_id, samples_per_label)
        for axis, index in zip(axes, indices):
            mappable = axis.imshow(
                contrast[index],
                cmap=_VARIANCE_CMAP,
                vmin=log10_floor,
                vmax=log10_ceiling,
            )
    if mappable is not None:
        all_axes = [axis for axes in slots.values() for axis in axes]
        colorbar = fig.colorbar(mappable, ax=all_axes, fraction=0.02, pad=0.01)
        colorbar.set_label("log10 normalized variance position", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    _save_png(fig, path)


def _label_grid(
    ordered_labels: list[int],
    samples_per_label: int,
) -> tuple[object, dict[int, list[object]]]:
    labels = [int(value) for value in ordered_labels]
    per_label = int(samples_per_label)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("ordered_labels must be non-empty and unique")
    if per_label < 1:
        raise ValueError("samples_per_label must be positive")

    if len(labels) <= 10:
        class_columns = 1
    elif per_label == 1:
        class_columns = min(5, len(labels))
    else:
        class_columns = 2
    rows = math.ceil(len(labels) / class_columns)
    columns = class_columns * per_label
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(max(1.2 * columns, 1.5), max(1.2 * rows, 1.5)),
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.03, hspace=0.18)
    for axis in axes.flat:
        axis.axis("off")

    slots: dict[int, list[object]] = {}
    for index, class_id in enumerate(labels):
        row = index // class_columns
        block = index % class_columns
        selected = list(axes[row, block * per_label : (block + 1) * per_label])
        selected[0].set_title(str(class_id), fontsize=7, pad=1)
        slots[class_id] = selected
    return fig, slots


def _class_indices(
    labels: np.ndarray,
    class_id: int,
    samples_per_label: int,
) -> np.ndarray:
    indices = np.flatnonzero(labels == int(class_id))
    if len(indices) != int(samples_per_label):
        raise ValueError(
            f"class {class_id} has {len(indices)} samples, expected {samples_per_label}"
        )
    return indices


def _labels_array(labels: torch.Tensor, *, expected_count: int) -> np.ndarray:
    values = labels.detach().cpu().numpy()
    if values.ndim != 1 or len(values) != int(expected_count):
        raise ValueError("labels must be one-dimensional and match the image count")
    return values


def _image_array(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().float().numpy()
    if array.ndim != 4 or array.shape[1] not in {1, 3}:
        raise ValueError("sample images must have shape [N, 1|3, H, W]")
    if array.shape[1] == 1:
        return array[:, 0]
    return np.moveaxis(array, 1, -1)


def _map_array(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().float().numpy()
    if array.ndim != 4 or array.shape[1] < 1:
        raise ValueError("variance maps must have shape [N, C, H, W]")
    return array[:, 0] if array.shape[1] == 1 else array.mean(axis=1)


def _save_png(fig, path: str | Path) -> None:
    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("sample visualization outputs must be PNG files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(output) as temporary:
        fig.savefig(
            temporary,
            format="png",
            dpi=_DPI,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
