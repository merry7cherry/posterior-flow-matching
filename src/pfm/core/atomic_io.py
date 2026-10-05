from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import torch


@contextmanager
def atomic_output_path(path: str | Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(file_descriptor)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_json_atomic(path: str | Path, payload: object) -> None:
    with atomic_output_path(path) as tmp_path:
        tmp_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_json_exclusive(path: str | Path, payload: object) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(file_descriptor)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        try:
            os.link(tmp_path, target)
        except FileExistsError:
            return False
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_torch_atomic(path: str | Path, payload: object) -> None:
    with atomic_output_path(path) as tmp_path:
        torch.save(payload, tmp_path)
