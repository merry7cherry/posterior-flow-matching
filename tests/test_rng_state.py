import random

import numpy as np
import pytest
import torch

from pfm.core.distributed import DistributedContext
from pfm.core.rng_context import preserve_rng_state
from pfm.core.rng_state import capture_rng_state, gather_rng_state_for_checkpoint, restore_rng_state, rng_state_for_rank
from pfm.core.rng_state import validate_rng_state_payload
from pfm.training.rng import rank_training_seed, seed_rank_training_rng


def _valid_rng_state(rank: int = 0):
    state = capture_rng_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rank)
    state["torch_cpu"] = generator.get_state()
    state["torch_cuda"] = None
    return state


def _numpy_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2] == right[2]
        and left[3] == right[3]
        and left[4] == right[4]
    )


def test_rank_training_seed_is_stable_and_rank_specific():
    assert rank_training_seed(1234, 0) == 1234
    assert rank_training_seed(1234, 1) != 1234
    assert rank_training_seed(1234, 1) == rank_training_seed(1234, 1)


def test_seed_rank_training_rng_changes_torch_stream_by_rank():
    seed_rank_training_rng(2026, rank=0)
    rank0 = torch.rand(4)
    seed_rank_training_rng(2026, rank=1)
    rank1 = torch.rand(4)
    assert not torch.equal(rank0, rank1)


def test_seed_rank_training_rng_seeds_python_numpy_and_torch():
    seed_rank_training_rng(99, rank=2)
    python_value = random.random()
    numpy_value = float(np.random.rand())
    torch_value = float(torch.rand(()))

    seed_rank_training_rng(99, rank=2)
    assert random.random() == python_value
    assert float(np.random.rand()) == numpy_value
    assert float(torch.rand(())) == torch_value


def test_rng_state_roundtrip_restores_python_numpy_and_torch():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    state = capture_rng_state()
    expected_python = [random.random() for _ in range(3)]
    expected_numpy = np.random.rand(3).tolist()
    expected_torch = torch.rand(3)

    restore_rng_state(state, require_cuda=False)

    assert [random.random() for _ in range(3)] == expected_python
    assert np.random.rand(3).tolist() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_preserve_rng_state_without_device_restores_python_numpy_and_torch():
    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(2)

    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    with preserve_rng_state():
        random.random()
        np.random.rand()
        torch.rand(2)

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(2), expected_torch)


def test_rng_state_schema_has_no_mps_field():
    state = capture_rng_state()

    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_single_process_gather_returns_one_rank_state():
    context = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )

    states = gather_rng_state_for_checkpoint(context)

    assert isinstance(states, list)
    assert len(states) == 1
    assert "torch_cpu" in states[0]


def test_rng_state_for_rank_requires_rng_state_by_rank():
    with pytest.raises(KeyError, match="rng_state_by_rank"):
        rng_state_for_rank({"rng_state": {"legacy": True}}, rank=0, world_size=1)


def test_rng_state_for_rank_rejects_non_list_rng_state_by_rank():
    with pytest.raises(ValueError, match="rng_state_by_rank"):
        rng_state_for_rank({"rng_state_by_rank": {"rank": 0}}, rank=0, world_size=1)


def test_rng_state_for_rank_rejects_wrong_world_size_length():
    with pytest.raises(ValueError, match="world size"):
        rng_state_for_rank({"rng_state_by_rank": [{"rank": 0}]}, rank=0, world_size=2)


def test_rng_state_for_rank_rejects_none_rank_state():
    with pytest.raises(ValueError, match="rank 0 RNG state must be a mapping"):
        rng_state_for_rank({"rng_state_by_rank": [None]}, rank=0, world_size=1)


def test_rng_state_for_rank_rejects_missing_required_keys():
    with pytest.raises(ValueError, match="rank 0 RNG state is missing fields"):
        rng_state_for_rank({"rng_state_by_rank": [{"python": random.getstate()}]}, rank=0, world_size=1)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "match"),
    [
        ("python", "not-a-python-rng-state", "python"),
        ("numpy", "not-a-numpy-rng-state", "numpy"),
        ("torch_cpu", torch.tensor([0], dtype=torch.int64), "torch_cpu"),
        ("torch_cpu", torch.zeros((1, 1), dtype=torch.uint8), "torch_cpu"),
        ("torch_cuda", "not-cuda-state-list", "torch_cuda"),
        ("torch_cuda", [], "non-empty"),
        ("torch_cuda", [torch.tensor([0], dtype=torch.int64)], "torch_cuda"),
        ("torch_cuda", [torch.zeros((1, 1), dtype=torch.uint8)], "torch_cuda"),
    ],
)
def test_rng_state_validation_rejects_malformed_values(field_name, invalid_value, match):
    state = _valid_rng_state()
    state[field_name] = invalid_value

    with pytest.raises(ValueError, match=match):
        validate_rng_state_payload(state, context="checkpoint")


def test_rng_state_validation_uses_isolated_validators_without_mutating_global_rng():
    state = _valid_rng_state()
    state["python"] = "not-a-python-rng-state"
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    with pytest.raises(ValueError, match="python"):
        validate_rng_state_payload(state, context="checkpoint")

    assert random.getstate() == python_state
    assert _numpy_states_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_rng_state_validation_rejects_non_restorable_one_byte_cpu_state():
    state = _valid_rng_state()
    state["torch_cpu"] = torch.tensor([0], dtype=torch.uint8)

    with pytest.raises(ValueError, match="torch_cpu.*not restorable"):
        validate_rng_state_payload(state, context="checkpoint")


def test_rng_state_validation_accepts_isolated_cpu_generator_state_without_mutating_global_rng():
    state = _valid_rng_state(rank=17)
    global_state = torch.random.get_rng_state()

    validated = validate_rng_state_payload(state, context="checkpoint")

    assert torch.equal(validated["torch_cpu"], state["torch_cpu"])
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_rng_state_restore_rejects_cuda_state_when_cuda_is_unavailable(monkeypatch):
    state = _valid_rng_state()
    state["torch_cuda"] = [_valid_rng_state(rank=1)["torch_cpu"]]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="torch_cuda.*CUDA is unavailable"):
        restore_rng_state(state, require_cuda=True)


def test_rng_state_restore_rejects_cuda_state_for_cpu_training():
    state = _valid_rng_state()
    state["torch_cuda"] = [_valid_rng_state(rank=1)["torch_cpu"]]

    with pytest.raises(ValueError, match="torch_cuda cannot be restored for CPU training"):
        restore_rng_state(state, require_cuda=False)


def test_rng_state_structural_validation_accepts_cuda_state_without_cuda_runtime(monkeypatch):
    state = _valid_rng_state()
    state["torch_cuda"] = [_valid_rng_state(rank=1)["torch_cpu"]]

    def unexpected_cuda_probe():
        raise AssertionError("structural validation must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_probe)

    validated = validate_rng_state_payload(
        state,
        context="aggregate checkpoint",
    )

    assert torch.equal(validated["torch_cuda"][0], state["torch_cuda"][0])


@pytest.mark.parametrize(
    "cuda_state",
    [
        torch.tensor([0], dtype=torch.int64),
        torch.zeros((1, 1), dtype=torch.uint8),
    ],
)
def test_rng_state_structural_validation_rejects_malformed_cuda_tensor(
    cuda_state,
    monkeypatch,
):
    state = _valid_rng_state()
    state["torch_cuda"] = [cuda_state]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match=r"torch_cuda\[0\].*1-D uint8"):
        validate_rng_state_payload(
            state,
            context="aggregate checkpoint",
        )


def test_rng_state_restore_rejects_incompatible_cuda_device_count(monkeypatch):
    state = _valid_rng_state()
    state["torch_cuda"] = [
        _valid_rng_state(rank=1)["torch_cpu"],
        _valid_rng_state(rank=2)["torch_cpu"],
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match=r"torch_cuda contains 2 device states.*1 CUDA devices"):
        restore_rng_state(state, require_cuda=True)


def test_rng_state_restore_rejects_non_restorable_cuda_state_on_cuda_runtime(monkeypatch):
    state = _valid_rng_state()
    state["torch_cuda"] = [torch.tensor([0], dtype=torch.uint8)]
    cpu_generator_factory = torch.Generator

    class RejectingCudaGenerator:
        def set_state(self, value):
            raise RuntimeError("invalid CUDA RNG state")

    def generator_factory(*, device):
        if device == "cpu":
            return cpu_generator_factory(device=device)
        assert device == "cuda:0"
        return RejectingCudaGenerator()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch, "Generator", generator_factory)

    with pytest.raises(ValueError, match=r"torch_cuda\[0\].*not restorable"):
        restore_rng_state(state, require_cuda=True)


def test_rng_state_restore_requires_cuda_state_for_cuda_training():
    with pytest.raises(ValueError, match="torch_cuda is required for CUDA training"):
        restore_rng_state(_valid_rng_state(), require_cuda=True)


def test_rng_state_for_rank_rejects_unknown_old_mps_field():
    state = _valid_rng_state()
    state["torch_mps"] = None

    with pytest.raises(ValueError, match="unknown fields"):
        rng_state_for_rank({"rng_state_by_rank": [state]}, rank=0, world_size=1)


def test_rng_state_for_rank_returns_current_rank_state():
    rank0 = _valid_rng_state(rank=0)
    rank1 = _valid_rng_state(rank=1)

    selected = rng_state_for_rank({"rng_state_by_rank": [rank0, rank1]}, rank=1, world_size=2)

    assert torch.equal(selected["torch_cpu"], rank1["torch_cpu"])
