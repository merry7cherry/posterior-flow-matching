import pytest
import yaml

from pfm.core.config import ExperimentConfig, experiment_config_from_mapping, load_config
from pfm.models.jit_config import JIT_MODEL_SPECS, jit_spec_for


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_production_jit_model_specs_exclude_test_only_models():
    assert set(JIT_MODEL_SPECS) == {
        "JiT-B/16",
        "JiT-L/16",
        "JiT-H/16",
    }
    assert "JiT-Tiny/4" not in JIT_MODEL_SPECS

    b16 = jit_spec_for("JiT-B/16", image_size=256)
    l16 = jit_spec_for("JiT-L/16", image_size=256)
    h16 = jit_spec_for("JiT-H/16", image_size=256)

    assert b16.patch_size == 16
    assert b16.depth < l16.depth
    assert h16.patch_size == 16
    assert h16.hidden_size > l16.hidden_size


def test_jit_spec_rejects_model_resolution_mismatch():
    with pytest.raises(ValueError, match="image_size"):
        jit_spec_for("JiT-B/16", image_size=512)


def test_in_memory_jit_l16_config_preserves_reference_scale():
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "imagenet256_jit_b16.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["jit_model"] = "JiT-L/16"
    config = experiment_config_from_mapping(raw)

    assert config.dataset == "imagenet"
    assert config.architecture == "jit_transformer"
    assert config.jit_model == "JiT-L/16"
    assert config.image_channels == 3
    assert config.image_size == 256
    assert config.num_classes == 1000
    assert config.noise_scale == 1.0
    assert not hasattr(config, "sampling_" + "method")


def test_imagenet256_jit_paper_config_uses_real_resolution_and_backbone():
    config = load_config(PROJECT_ROOT / "configs" / "imagenet256_jit_b16_batch256.yaml")

    assert config.dataset == "imagenet"
    assert config.image_size == 256
    assert config.jit_model == "JiT-B/16"
    assert config.batch_size == 256
    assert config.max_grad_norm == 5.0
    assert not hasattr(config, "metric_names")


def test_general_config_preserves_existing_noise_default():
    config = ExperimentConfig()

    assert config.noise_scale == 1.0
    assert not hasattr(config, "sampling_" + "method")
