from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from pfm.evaluation import fid_metrics


def test_explicit_weight_preparation_is_reusable_and_records_full_digest(tmp_path, monkeypatch):
    payload = b"fixture Inception weights"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(fid_metrics, "_TORCH_FIDELITY_INCEPTION_SHA256_PREFIX", digest[:8])
    calls = []

    def download(url, destination, *, hash_prefix, progress):
        calls.append(url)
        assert hash_prefix == digest[:8]
        Path(destination).write_bytes(payload)

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    first = fid_metrics.prepare_inception_weights()
    assert first["sha256"] == digest
    assert Path(first["path"]).read_bytes() == payload
    assert fid_metrics.prepare_inception_weights() == first
    assert len(calls) == 1
    assert not list((tmp_path / "checkpoints").glob(".inception-*"))


def test_interrupted_weight_download_does_not_publish_partial_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    def download(url, destination, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("interrupted download")

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    with pytest.raises(RuntimeError, match="interrupted"):
        fid_metrics.prepare_inception_weights()
    assert list((tmp_path / "checkpoints").iterdir()) == []


def test_existing_corrupt_weight_is_rejected_without_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    weights = tmp_path / "checkpoints" / fid_metrics._TORCH_FIDELITY_INCEPTION_WEIGHTS_NAME
    weights.parent.mkdir()
    weights.write_bytes(b"corrupt")
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("download"))
    with pytest.raises(ValueError, match="digest mismatch"):
        fid_metrics.prepare_inception_weights()
    assert weights.read_bytes() == b"corrupt"
