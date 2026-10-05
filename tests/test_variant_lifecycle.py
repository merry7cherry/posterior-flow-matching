from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

import pfm.core.variant_lifecycle as lifecycle_module
from pfm.core.variant_lifecycle import variant_writer_lock


PROCESS_TIMEOUT_SECONDS = 10.0


def _hold_writer(variant_root: str, acquired, release) -> None:
    with variant_writer_lock(Path(variant_root)):
        acquired.set()
        if not release.wait(PROCESS_TIMEOUT_SECONDS):
            raise TimeoutError("writer release event was not set")


def _attempt_writer(variant_root: str, result) -> None:
    try:
        with variant_writer_lock(Path(variant_root)):
            result.put("acquired")
    except RuntimeError as error:
        result.put(str(error))


def _join(process: multiprocessing.Process) -> None:
    process.join(PROCESS_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(PROCESS_TIMEOUT_SECONDS)
        pytest.fail("writer process did not exit")
    assert process.exitcode == 0


def test_second_writer_for_the_same_variant_fails_without_waiting(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    variant_root = tmp_path / "task" / "variant"
    acquired = context.Event()
    release = context.Event()
    result = context.Queue()
    holder = context.Process(target=_hold_writer, args=(str(variant_root), acquired, release))
    contender = context.Process(target=_attempt_writer, args=(str(variant_root), result))

    holder.start()
    try:
        assert acquired.wait(PROCESS_TIMEOUT_SECONDS)
        contender.start()
        _join(contender)
        assert "already owns variant" in result.get(timeout=PROCESS_TIMEOUT_SECONDS)
    finally:
        release.set()
        _join(holder)


def test_writer_lock_is_task_scoped_before_variant_exists(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    variant_root = task_root / "variant"

    with variant_writer_lock(variant_root):
        assert (task_root / ".variant.writer.lock").is_file()
        assert not variant_root.exists()


def test_writer_lock_rejects_missing_variant_component() -> None:
    with pytest.raises(ValueError, match="component"):
        with variant_writer_lock(Path("/")):
            pytest.fail("invalid root must fail")


@pytest.mark.parametrize("flag_name", ["O_CLOEXEC", "O_NOFOLLOW"])
def test_writer_lock_requires_linux_open_flags(
    tmp_path: Path,
    monkeypatch,
    flag_name: str,
) -> None:
    os_module = lifecycle_module.os

    class OSWithoutFlag:
        def __getattr__(self, name):
            if name == flag_name:
                raise AttributeError(name)
            return getattr(os_module, name)

    monkeypatch.setattr(lifecycle_module, "os", OSWithoutFlag())

    with pytest.raises(AttributeError, match=flag_name):
        with variant_writer_lock(tmp_path / "task" / "variant"):
            pytest.fail("missing required Linux open flag must fail")
