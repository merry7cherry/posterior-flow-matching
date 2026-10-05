from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pfm.core.layout import validate_path_component
from pfm.core.execution import require_canonical_path


@contextmanager
def variant_writer_lock(variant_root: str | Path) -> Iterator[None]:
    """Exclude a second trainer without blocking or locking immutable readers."""
    raw_variant_root = Path(variant_root)
    validate_path_component(raw_variant_root.name, "variant_id")
    lexical_variant_root = Path(os.path.abspath(raw_variant_root))
    task_root = lexical_variant_root.parent
    validate_path_component(task_root.name, "task_id")
    if task_root.is_symlink():
        raise ValueError(f"task_root must not be a symlink: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)
    canonical_variant_root = require_canonical_path(
        lexical_variant_root,
        task_root,
        name="variant_root",
    )
    variant_id = canonical_variant_root.name
    lock_path = task_root / f".{variant_id}.writer.lock"
    if lock_path.is_symlink():
        raise ValueError(f"writer lock path must not be a symlink: {lock_path}")

    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another trainer already owns variant {variant_id!r}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
