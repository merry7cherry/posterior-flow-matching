from pathlib import Path

import pytest
import yaml

import pfm.core.config as config_module
from pfm.core.config import ExperimentConfig, load_config
from pfm.core.evaluation_config import (
    load_metric_evaluation_config,
    metric_evaluation_config_from_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_name,field,value", [
    ("mnist_unet.yaml", "num_classes", 5),
    ("mnist_unet.yaml", "image_channels", 3),
    ("mnist_unet.yaml", "image_size", 32),
    ("cifar10_edm_ddpmpp.yaml", "num_classes", 5),
    ("cifar10_edm_ddpmpp.yaml", "image_channels", 1),
    ("cifar10_edm_ddpmpp.yaml", "image_size", 28),
    ("imagenet256_jit_b16_batch256.yaml", "num_classes", 10),
    ("imagenet256_jit_b16_batch256.yaml", "image_channels", 1),
])
def test_fixed_dataset_attributes_reject_yaml_override_and_checkpoint(config_name, field, value):
    path = PROJECT_ROOT / "configs" / config_name
    raw = yaml.safe_load(path.read_text())
    raw[field] = value
    with pytest.raises(ValueError, match=f"requires {field}="):
        config_module.experiment_config_from_mapping(raw)
    with pytest.raises(ValueError, match=f"requires {field}="):
        load_config(path, overrides={field: value})


def test_fixed_dataset_attributes_preserve_matching_values_and_tuning():
    config = load_config(PROJECT_ROOT / "configs/cifar10_edm_ddpmpp.yaml",
                         overrides={"num_classes": 10, "image_channels": 3,
                                    "seed": 19, "batch_size": 64, "lr": 0.001})
    assert (config.seed, config.batch_size, config.lr) == (19, 64, 0.001)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/cifar10_edm_ddpmpp.yaml",
        "configs/mnist_unet.yaml",
    ],
)
def test_baseline_configs_restore_approved_cuda_numerical_protocol(config_path):
    config = load_config(PROJECT_ROOT / config_path)
    assert config.mixed_precision == "bf16"
    assert config.allow_tf32 is False
    assert config.cudnn_benchmark is True
    assert config.drop_last is True


@pytest.mark.parametrize("field", ["device", "data_root", "output_root"])
def test_experiment_config_rejects_retired_runtime_fields(tmp_path, field):
    path = tmp_path / "config.yaml"
    path.write_text(f"{field}: value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(path)


@pytest.mark.parametrize(
    "field",
    ["device", "data_root", "classifier_checkpoint", "fid_real_stats", "fid_cache_dir"],
)
def test_evaluation_config_rejects_retired_runtime_fields(tmp_path, field):
    path = tmp_path / "evaluation.yaml"
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs/evaluation/mnist_generation.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw[field] = "value"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        load_metric_evaluation_config(path)


@pytest.mark.parametrize(
    "config_name",
    ["mnist_generation.yaml", "mnist_posterior.yaml"],
)
def test_evaluation_config_rejects_retired_smoke_field(config_name):
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "evaluation" / config_name).read_text(
            encoding="utf-8"
        )
    )
    raw["smoke"] = False

    with pytest.raises(ValueError, match="keys"):
        metric_evaluation_config_from_mapping(raw)


MNIST_CONFIG_PATH = PROJECT_ROOT / "configs" / "mnist_unet.yaml"


def _write_mnist_config(path: Path, updates: dict[str, object]) -> None:
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.update(updates)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("config_text", ["", "# empty document\n", "{}\n"])
def test_load_config_rejects_empty_authored_config(tmp_path, config_text):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="required authored config keys"):
        config_module.load_config(config_path)


@pytest.mark.parametrize("config_text", ["[]\n", "false\n", "0\n", "- dataset\n"])
def test_load_config_rejects_non_mapping_yaml_documents(tmp_path, config_text):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="YAML document must be a mapping"):
        config_module.load_config(config_path)


@pytest.mark.parametrize("config_text", ["", "# empty document\n", "{}\n"])
def test_load_metric_evaluation_config_rejects_empty_authored_config(tmp_path, config_text):
    config_path = tmp_path / "empty_eval.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="profile is required"):
        load_metric_evaluation_config(config_path)


@pytest.mark.parametrize("config_text", ["[]\n", "false\n", "0\n", "- generation\n"])
def test_load_metric_evaluation_config_rejects_non_mapping_yaml_documents(tmp_path, config_text):
    config_path = tmp_path / "invalid_eval.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="YAML document must be a mapping"):
        load_metric_evaluation_config(config_path)


@pytest.mark.parametrize("loader", [config_module.load_config, load_metric_evaluation_config])
def test_authored_yaml_rejects_duplicate_keys_before_coercion(tmp_path, loader):
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text("dataset: mnist\ndataset: cifar10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate.*dataset"):
        loader(config_path)


def test_mnist_unet_config_loads_unet_defaults():
    assert MNIST_CONFIG_PATH.exists()

    config = config_module.load_config(MNIST_CONFIG_PATH)

    assert config.architecture == "conditional_unet"
    assert config.image_channels == 1
    assert config.image_size == 28
    assert config.time_proj_dim == 128
    assert config.class_proj_dim == 128
    assert config.unet_base_channels == 32
    assert config.unet_channel_mult == (1, 2, 4)
    assert config.unet_num_res_blocks == 2
    assert config.unet_attention_resolutions == ()
    assert config.unet_num_heads == 1
    assert config.unet_dropout == 0.05
    assert config.time_embed_scale == 1000.0


def test_load_config_reads_exact_supplied_path_from_non_project_cwd(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MNIST_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = config_module.load_config(config_path)

    assert config.dataset == "mnist"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jit_compile", "false"),
        ("allow_tf32", 1),
        ("self_conditioning_mode", True),
        ("batch_size", "256"),
        ("lr", "0.0002"),
        ("lr", "none"),
        ("unet_channel_mult", ["1", "2"]),
    ],
)
def test_experiment_config_rejects_non_native_scalar_and_sequence_types(field, value):
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw[field] = value

    with pytest.raises(ValueError, match=field):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("config_name", "field"),
    [
        ("mnist_unet.yaml", "lr"),
        ("imagenet256_jit_b16.yaml", "base_lr"),
        ("mnist_unet.yaml", "time_embed_scale"),
        ("cifar10_edm_ddpmpp.yaml", "edm_dropout"),
        ("imagenet256_jit_b16.yaml", "jit_attn_dropout"),
        ("mnist_unet.yaml", "weight_decay"),
        ("mnist_unet.yaml", "ema_decay"),
        ("mnist_unet.yaml", "eps_t"),
        ("mnist_unet.yaml", "logit_normal_p_mean"),
        ("mnist_unet.yaml", "logit_normal_p_std"),
        ("mnist_unet.yaml", "logvar_min"),
        ("mnist_unet.yaml", "logvar_max"),
        ("mnist_unet.yaml", "logvar_init"),
        ("mnist_unet.yaml", "noise_scale"),
        ("mnist_unet.yaml", "adamw_beta1"),
        ("mnist_unet.yaml", "adamw_beta2"),
        ("mnist_unet.yaml", "max_grad_norm"),
    ],
)
def test_experiment_config_rejects_non_finite_authored_numbers(config_name, field, value):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    raw[field] = value

    with pytest.raises(ValueError, match=f"{field}.*finite"):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize(
    ("config_name", "field", "value", "message"),
    [
        ("mnist_unet.yaml", "num_classes", 0, "num_classes"),
        ("mnist_unet.yaml", "image_channels", 0, "image_channels"),
        ("mnist_unet.yaml", "image_size", 0, "image_size"),
        ("mnist_unet.yaml", "batch_size", 0, "batch_size"),
        ("mnist_unet.yaml", "num_workers", -1, "num_workers"),
        ("mnist_unet.yaml", "epochs", 0, "epochs"),
        ("mnist_unet.yaml", "weight_decay", -1.0, "weight_decay"),
        ("mnist_unet.yaml", "ema_decay", 1.2, "ema_decay"),
        ("mnist_unet.yaml", "noise_scale", 0.0, "noise_scale"),
        ("mnist_unet.yaml", "logvar_min", 2.0, "logvar"),
        ("mnist_unet.yaml", "logvar_init", 2.0, "logvar"),
        ("mnist_unet.yaml", "unet_channel_mult", [], "unet_channel_mult"),
        ("mnist_unet.yaml", "unet_channel_mult", [1, 0], "unet_channel_mult"),
        ("mnist_unet.yaml", "unet_norm_groups", 0, "unet_norm_groups"),
        ("cifar10_edm_ddpmpp.yaml", "edm_dropout", 1.0, "edm_dropout"),
        ("cifar10_edm_ddpmpp.yaml", "edm_channel_mult", [], "edm_channel_mult"),
        ("imagenet256_jit_b16.yaml", "jit_attn_dropout", -0.1, "jit_attn_dropout"),
    ],
)
def test_experiment_config_rejects_invalid_scientific_or_runtime_ranges(
    config_name,
    field,
    value,
    message,
):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize("yaml_value", [".nan", ".inf", "-.inf"])
def test_experiment_yaml_rejects_non_finite_lr(tmp_path, yaml_value):
    raw = MNIST_CONFIG_PATH.read_text(encoding="utf-8").replace("lr: 0.0002", f"lr: {yaml_value}")
    path = tmp_path / "nonfinite.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="lr.*finite"):
        config_module.load_config(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_evaluation_config_rejects_non_finite_posterior_times(value):
    path = PROJECT_ROOT / "configs" / "evaluation" / "cifar10_edm_posterior.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["posterior_t_values"][2] = value

    with pytest.raises(ValueError, match="posterior_t_values.*finite"):
        metric_evaluation_config_from_mapping(raw)


@pytest.mark.parametrize("field", ["dataset", "architecture", "image_channels", "batch_size", "lr"])
def test_experiment_config_requires_authored_common_keys(field):
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.pop(field)

    with pytest.raises(ValueError, match=field):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize("field", ["unet_base_channels", "unet_channel_mult", "unet_norm_groups"])
def test_experiment_config_requires_selected_architecture_keys(field):
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.pop(field)

    with pytest.raises(ValueError, match=field):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", " MNIST"),
        ("dataset", "MNIST"),
        ("architecture", "Conditional_UNet"),
        ("architecture", "conditional_unet "),
    ],
)
def test_experiment_config_rejects_noncanonical_identity(field, value):
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw[field] = value

    with pytest.raises(ValueError, match=field):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize("value", ["Uniform", " uniform", "uniform ", "logit-normal"])
def test_experiment_config_rejects_noncanonical_time_sampling(value):
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["time_sampling"] = value

    with pytest.raises(ValueError, match="time_sampling"):
        config_module.experiment_config_from_mapping(raw)


def test_evaluation_config_rejects_scalar_metric_names():
    path = PROJECT_ROOT / "configs" / "evaluation" / "cifar10_edm_generation.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["metric_names"] = "generation"

    with pytest.raises(ValueError, match="metric_names must be a sequence"):
        metric_evaluation_config_from_mapping(raw)


def test_experiment_config_from_mapping_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.experiment_config_from_mapping({"metric_names": ["generation"]})


@pytest.mark.parametrize(
    ("config_name", "inactive_field"),
    [
        ("mnist_unet.yaml", "edm_model_channels"),
        ("cifar10_edm_ddpmpp.yaml", "unet_base_channels"),
        ("imagenet256_jit_b16.yaml", "unet_dropout"),
    ],
)
def test_experiment_config_rejects_fields_owned_by_other_architectures(
    config_name,
    inactive_field,
):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    raw[inactive_field] = getattr(ExperimentConfig(), inactive_field)

    with pytest.raises(ValueError, match=f"architecture.*{inactive_field}"):
        config_module.experiment_config_from_mapping(raw)


@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_unet.yaml",
        "cifar10_edm_ddpmpp.yaml",
        "imagenet256_jit_b16.yaml",
        "imagenet256_jit_b16_batch256.yaml",
    ],
)
def test_experiment_config_serializer_round_trips_exact_architecture_schema(config_name):
    config = config_module.load_config(PROJECT_ROOT / "configs" / config_name)

    payload = config_module.experiment_config_to_mapping(config)

    assert set(payload) == (
        config_module.COMMON_REQUIRED_KEYS
        | config_module.ARCHITECTURE_REQUIRED_KEYS[config.architecture]
    )
    assert config_module.experiment_config_from_mapping(payload) == config


def test_shared_training_protocol_excludes_segment_schedule_but_not_lr_authorship():
    from dataclasses import replace

    config = config_module.load_config(MNIST_CONFIG_PATH)
    changed_segment = replace(config, epochs=config.epochs + 1, checkpoint_every_epochs=7)
    changed_lr = replace(config, lr=config.lr * 2)

    assert config_module.shared_training_protocol_sha256(changed_segment) == (
        config_module.shared_training_protocol_sha256(config)
    )
    assert config_module.shared_training_protocol_sha256(changed_lr) != (
        config_module.shared_training_protocol_sha256(config)
    )


@pytest.mark.parametrize("value", [None, 1, 2])
def test_training_batch_limit_is_recorded_in_config_and_protocol(value):
    config = load_config(MNIST_CONFIG_PATH, overrides={"limit_train_batches": value})
    payload = config_module.experiment_config_to_mapping(config)
    assert payload["limit_train_batches"] == value
    assert config_module.experiment_config_from_mapping(payload) == config
    baseline = load_config(MNIST_CONFIG_PATH)
    assert (config_module.shared_training_protocol_sha256(config) ==
            config_module.shared_training_protocol_sha256(baseline)) == (value is None)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
def test_training_batch_limit_rejects_nonpositive_or_noninteger_values(value):
    with pytest.raises(ValueError, match="limit_train_batches"):
        load_config(MNIST_CONFIG_PATH, overrides={"limit_train_batches": value})


def test_training_batch_limit_is_required_in_authored_config():
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text())
    del raw["limit_train_batches"]
    with pytest.raises(ValueError, match="required authored config keys.*limit_train_batches"):
        config_module.experiment_config_from_mapping(raw)


def test_bundled_edm_config_has_no_inactive_unet_fields():
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "cifar10_edm_ddpmpp.yaml").read_text(encoding="utf-8")
    )

    assert set(raw) == (
        config_module.COMMON_REQUIRED_KEYS
        | config_module.ARCHITECTURE_REQUIRED_KEYS["edm_ddpmpp"]
    )


def test_external_config_rejects_retired_relative_path_fields(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "external.yaml"
    config_path.write_text("data_root: datasets\noutput_root: runs\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(config_path)


def test_in_memory_cifar10_unet_config_supports_rgb_geometry():
    raw = yaml.safe_load(MNIST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw.update(
        {
            "dataset": "cifar10",
            "image_channels": 3,
            "image_size": 32,
        }
    )
    config = config_module.experiment_config_from_mapping(raw)

    assert config.dataset == "cifar10"
    assert config.architecture == "conditional_unet"
    assert config.image_channels == 3
    assert config.image_size == 32
    assert config.num_classes == 10


def test_default_variance_range_is_tight_for_mnist_toy():
    config = config_module.load_config(MNIST_CONFIG_PATH)

    assert config.logvar_min == -1.0
    assert config.logvar_max == 1.0
    assert ExperimentConfig().logvar_min == -1.0
    assert ExperimentConfig().logvar_max == 1.0
    assert config.epochs == 200


def test_config_has_no_covariance_selection_fields():
    config = ExperimentConfig()

    for suffix in ("type", "rank", "jitter", "factor_init_fraction"):
        field_name = "covariance_" + suffix
        assert not hasattr(config, field_name)


def test_config_has_no_metric_nfe_field():
    config = ExperimentConfig()

    assert not hasattr(config, "metric_nfe")


def test_config_has_no_sampler_method_field():
    config = ExperimentConfig()
    stale_key = "sampling_" + "method"

    assert stale_key not in ExperimentConfig.__dataclass_fields__
    assert not hasattr(config, stale_key)


def test_self_conditioning_defaults_are_disabled():
    config = ExperimentConfig()

    assert config.self_conditioning_mode == "none"
    for old_field in (
        "self_conditioning",
        "self_conditioning_prob",
        "self_conditioning_use_logvar",
        "self_conditioning_sampling",
        "self_conditioning_sample_use_logvar",
    ):
        assert old_field not in ExperimentConfig.__dataclass_fields__


def test_config_rejects_stale_sampler_method_option(tmp_path):
    config_path = tmp_path / "stale.yaml"
    stale_key = "sampling_" + "method"
    stale_value = "he" + "un"
    config_path.write_text(f"dataset: mnist\n{stale_key}: {stale_value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(config_path)


def test_config_rejects_stale_covariance_options(tmp_path):
    config_path = tmp_path / "stale.yaml"
    config_path.write_text(
        "dataset: mnist\n"
        f"covariance_{'type'}: removed\n"
        f"covariance_{'rank'}: 8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(config_path)


def test_config_rejects_stale_metric_nfe_option(tmp_path):
    config_path = tmp_path / "stale.yaml"
    config_path.write_text("dataset: mnist\nmetric_nfe: 200\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(config_path)


def test_config_rejects_stale_self_conditioning_sample_logvar_option(tmp_path):
    config_path = tmp_path / "stale.yaml"
    config_path.write_text("dataset: mnist\nself_conditioning_sample_use_logvar: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(config_path)


def test_self_conditioning_config_rejects_retired_modes(tmp_path):
    for mode in ("mean", "mean_logvar"):
        config_path = tmp_path / f"{mode}.yaml"
        _write_mnist_config(config_path, {"self_conditioning_mode": mode})
        with pytest.raises(ValueError, match="self-conditioning runtime is retired"):
            config_module.load_config(config_path)


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        ("self_conditioning_mode: true\n", "self_conditioning_mode"),
        ("self_conditioning_mode: mean-plus-logvar\n", "self_conditioning_mode"),
    ],
)
def test_self_conditioning_config_rejects_invalid_values(tmp_path, config_text, message):
    config_path = tmp_path / "invalid.yaml"
    _write_mnist_config(config_path, yaml.safe_load(config_text))

    with pytest.raises(ValueError, match=message):
        config_module.load_config(config_path)


def test_default_optimizer_and_sampling_grid_match_mnist_toy_sweep():
    config = config_module.load_config(MNIST_CONFIG_PATH)

    assert config.lr == 2e-4
    assert config.base_lr is None
    assert config.lr_reference_batch_size == 256
    assert config.scale_lr_by_effective_batch is False
    assert ExperimentConfig().lr == 2e-4
    assert ExperimentConfig().base_lr is None
    assert ExperimentConfig().scale_lr_by_effective_batch is False


def test_config_loads_jit_style_effective_batch_lr_scaling(tmp_path):
    config_path = tmp_path / "jit.yaml"
    _write_mnist_config(
        config_path,
        {
            "lr": None,
            "base_lr": 0.00005,
            "lr_reference_batch_size": 256,
            "scale_lr_by_effective_batch": True,
        },
    )

    config = config_module.load_config(config_path)

    assert config.lr is None
    assert config.base_lr == 5e-5
    assert config.lr_reference_batch_size == 256
    assert config.scale_lr_by_effective_batch is True


def test_config_rejects_null_lr_without_effective_batch_scaling(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    _write_mnist_config(config_path, {"lr": None})

    with pytest.raises(ValueError, match="lr is required unless scale_lr_by_effective_batch"):
        config_module.load_config(config_path)


def test_training_config_has_no_evaluation_or_sampling_fields():
    removed_fields = {
        "nfe_values",
        "num_vis_per_class",
        "sample_every_epochs",
        "sample_seed",
        "metric_names",
        "metric_every_epochs",
        "metric_classifier_checkpoint",
        "generation_sampling_step_values",
        "metric_num_samples_per_class",
        "metric_num_calibration_batches",
        "metric_num_bins",
        "metric_batch_size",
        "posterior_t_values",
        "metric_fid_backend",
        "metric_fid_num_samples",
        "metric_fid_batch_size",
        "metric_fid_real_stats",
        "metric_fid_cache_dir",
        "metric_seed",
    }

    for field_name in removed_fields:
        assert field_name not in ExperimentConfig.__dataclass_fields__
        assert not hasattr(ExperimentConfig(), field_name)


def test_bundled_training_configs_do_not_contain_evaluation_or_sampling_keys():
    config_names = [
        "mnist_unet.yaml",
        "cifar10_edm_ddpmpp.yaml",
        "cifar10_edm_ddpmpp_logvar_neg2.yaml",
        "cifar10_edm_ddpmpp_logvar_neg3.yaml",
        "imagenet256_jit_b16.yaml",
        "imagenet256_jit_b16_batch256.yaml",
    ]
    actual_config_names = sorted(path.name for path in (PROJECT_ROOT / "configs").glob("*.yaml") if path.name != "resources.yaml")
    assert actual_config_names == sorted(config_names)

    forbidden = {
        "device",
        "data_root",
        "output_root",
        "nfe_values",
        "num_vis_per_class",
        "sample_every_epochs",
        "sample_seed",
        "metric_names",
        "metric_every_epochs",
        "metric_classifier_checkpoint",
        "generation_sampling_step_values",
        "metric_num_samples_per_class",
        "metric_num_calibration_batches",
        "metric_num_bins",
        "metric_batch_size",
        "posterior_t_values",
        "metric_fid_backend",
        "metric_fid_num_samples",
        "metric_fid_batch_size",
        "metric_fid_real_stats",
        "metric_fid_cache_dir",
        "metric_seed",
    }
    for name in config_names:
        text = (PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8")
        for key in forbidden:
            assert f"{key}:" not in text
        config = config_module.load_config(PROJECT_ROOT / "configs" / name)
        assert config.limit_train_batches is None


def test_training_config_rejects_evaluation_or_sampling_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "metric_names: [generation]",
                "sample_every_epochs: 10",
                "generation_sampling_step_values: [1, 2]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown config keys"):
        config_module.load_config(config_path)


def test_default_time_sampling_keeps_existing_uniform_behavior():
    config = config_module.load_config(MNIST_CONFIG_PATH)

    assert config.time_sampling == "uniform"
    assert ExperimentConfig().time_sampling == "uniform"
    assert config.logit_normal_p_mean == -0.8
    assert config.logit_normal_p_std == 0.8
    assert ExperimentConfig().logit_normal_p_mean == -0.8
    assert ExperimentConfig().logit_normal_p_std == 0.8


@pytest.mark.parametrize(
    ("config_name", "field", "value", "message"),
    [
        ("mnist_unet.yaml", "eps_t", 0.01, "eps_t=0.001"),
        ("cifar10_edm_ddpmpp.yaml", "time_sampling", "logit_normal", "uniform"),
        ("imagenet256_jit_b16.yaml", "time_sampling", "uniform", "logit-normal"),
        ("imagenet256_jit_b16.yaml", "logit_normal_p_mean", 0.0, "mean=-0.8"),
    ],
)
def test_training_time_parameters_are_configurable(
    tmp_path,
    config_name,
    field,
    value,
    message,
):
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8")
    )
    raw[field] = value
    path = tmp_path / config_name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert getattr(config_module.load_config(path), field) == value


def test_experiment_config_has_no_inactive_mlp_fields():
    stale_fields = {"hidden_dim", "num_hidden_layers", "input_dim"}

    assert stale_fields.isdisjoint(ExperimentConfig.__dataclass_fields__)


def test_user_facing_files_do_not_use_stale_mlp_names():
    checked_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "src" / "pfm" / "cli" / "train.py",
        PROJECT_ROOT / "src" / "pfm" / "models" / "factory.py",
    ]
    stale_tokens = {"mnist_mlp", "ConditionalMLP"}

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for token in stale_tokens:
            assert token not in text


@pytest.mark.parametrize("value", ["none", "bf16"])
def test_config_accepts_only_canonical_mixed_precision_values(tmp_path, value):
    config_path = tmp_path / "mixed_precision.yaml"
    _write_mnist_config(config_path, {"mixed_precision": value})

    config = config_module.load_config(config_path)

    assert config.mixed_precision == value


def test_mixed_precision_modes_define_the_complete_contract():
    assert config_module.MIXED_PRECISION_MODES == ("none", "bf16")


@pytest.mark.parametrize(
    "yaml_value",
    [
        '""',
        "off",
        '"false"',
        "fp32",
        "float32",
        "bfloat16",
        "BF16",
        '" bf16 "',
        "false",
        "null",
    ],
)
def test_config_rejects_noncanonical_mixed_precision_values(tmp_path, yaml_value):
    config_path = tmp_path / "invalid_mixed_precision.yaml"
    _write_mnist_config(config_path, {"mixed_precision": yaml.safe_load(yaml_value)})

    with pytest.raises(ValueError, match="mixed_precision must be one of"):
        config_module.load_config(config_path)
