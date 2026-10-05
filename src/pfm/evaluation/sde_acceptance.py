"""Bounded checkpoint-based PFM SDE accuracy/cost acceptance on allocated CUDA.

This script does not submit jobs or train models. It uses the canonical EMA
checkpoint reader and generation service. Coarse/fine comparisons explicitly
couple a test-only Brownian driver; they do not alter the production RNG API.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import asdict
from itertools import combinations
import hashlib
import json
import math
import time
from unittest.mock import patch

import torch

from pfm.core.immutable_artifact import prepare_immutable_directory, publish_immutable_directory
from pfm.core.layout import validate_path_component
from pfm.core.runtime import execution_device
from pfm.core.sampling_config import SamplingConfig
from pfm.core.variant_metadata import requires_learned_variance
from pfm.core.execution import require_canonical_path, runtime_identity
from pfm.data.datasets import make_datasets
from pfm.diagnostics.posterior_residuals import posterior_generator
from pfm.evaluation.checkpoint_loading import (
    load_model_from_bound_checkpoint,
    require_canonical_variant_root,
    resolve_bound_epoch_checkpoint,
    validate_bound_checkpoint_dataset,
)
from pfm.evaluation.generation_service import GenerationRequest, iter_loaded_generation
from pfm.sampling import samplers
from pfm.sampling.derivatives import estimate_variance_diagonal
from pfm.sampling.randomness import RANDOM_PROTOCOL, indexed_step_random


def coupled_step_random(
    sample_indices, sample_shape, *, seed, step, stream, probe=0, device,
    refinement: int,
):
    """Normalized fine Gaussian sums give the identical coarse Brownian path."""
    if type(refinement) is not int or refinement < 1:
        raise ValueError("refinement must be a positive integer")
    if stream != "brownian":
        return indexed_step_random(
            sample_indices, sample_shape, seed=seed, step=step * refinement,
            stream=stream, probe=probe, device=device,
        )
    total = torch.zeros((len(sample_indices), *sample_shape), device=device)
    for offset in range(refinement):
        total += indexed_step_random(
            sample_indices, sample_shape, seed=seed, step=step * refinement + offset,
            stream=stream, probe=probe, device=device,
        )
    return total / math.sqrt(refinement)


@contextmanager
def coupled_brownian_driver(steps: int, *, finest: int):
    if finest % steps:
        raise ValueError("comparison steps must divide the finest Brownian grid")

    def driver(*args, **kwargs):
        return coupled_step_random(*args, **kwargs, refinement=finest // steps)

    # Scoped strictly to these validation calls; public sampler remains intact.
    with patch.object(samplers, "indexed_step_random", driver):
        yield


def independent_group_error(first: torch.Tensor, second: torch.Tensor) -> float:
    """Per-component MSE estimate for either independent equal-budget estimator."""
    return float((first.double() - second.double()).square().mean() / 2.0)


def _cuda_measure(device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _cuda_result(started, device):
    torch.cuda.synchronize(device)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _estimate(model, variant, config, state, labels, *, t_value, precision, count, seed, group, device):
    current = state.detach().requires_grad_(True)
    times = torch.full((len(labels), 1, 1, 1), t_value, device=device)
    indices = torch.arange(len(labels), dtype=torch.long)
    with torch.enable_grad():
        prediction = samplers.predict_velocity_and_uncertainty(
            model, variant, current, times, labels, config, precision=precision,
        )
        gamma = prediction.velocity_var
        if gamma is None:
            raise ValueError("probe acceptance requires a PFM variance head")
        probes = (
            indexed_step_random(
                indices, tuple(current.shape[1:]), seed=seed, step=group,
                stream="derivative", probe=probe, device=device,
            )
            for probe in range(count)
        )
        estimate = estimate_variance_diagonal(gamma, current, probes, count)
    if not bool(torch.isfinite(estimate.value).all() & torch.isfinite(gamma).all() & (gamma > 0).all()):
        raise FloatingPointError("nonfinite derivative or invalid variance in probe benchmark")
    return estimate, gamma.detach()


def probe_benchmark(model, variant, config, x0, x1, labels, args, device):
    results = []
    estimates_for_precision_comparison = {}
    for precision in args.precisions:
        # Warm up that execution path separately; timing excludes the warmup.
        _estimate(
            model, variant, config, 0.5 * (x0 + x1), labels, t_value=0.5,
            precision=precision, count=2, seed=args.seed, group=9999, device=device,
        )
        for t_index, t_value in enumerate(args.reference_times):
            state = (1.0 - t_value) * x0 + t_value * x1
            for count in args.probe_counts:
                started = _cuda_measure(device)
                squared_errors = []
                internal_errors = []
                variance_means = []
                derivative_means = []
                actual_vjp = 0
                for repeat in range(args.probe_repeats):
                    # Group identity is independent of precision and m, so m
                    # comparisons use nested prefixes without changing x0/x1.
                    group = 100 * t_index + 2 * repeat
                    first, gamma = _estimate(
                        model, variant, config, state, labels, t_value=t_value,
                        precision=precision, count=count, seed=args.seed,
                        group=group, device=device,
                    )
                    second, _ = _estimate(
                        model, variant, config, state, labels, t_value=t_value,
                        precision=precision, count=count, seed=args.seed,
                        group=group + 1, device=device,
                    )
                    actual_vjp += first.actual_vjp + second.actual_vjp
                    squared_errors.append(independent_group_error(first.value, second.value))
                    variance_means.append(float(gamma.double().mean()))
                    derivative_means.append(float(first.value.double().square().mean()))
                    if first.mean_squared_error_estimate is not None:
                        internal_errors.extend([
                            float(first.mean_squared_error_estimate),
                            float(second.mean_squared_error_estimate),
                        ])
                    if repeat == 0 and count == max(args.probe_counts):
                        estimates_for_precision_comparison[(precision, t_value)] = first.value.cpu()
                    del first, second, gamma
                measurements = _cuda_result(started, device)
                error = sum(squared_errors) / len(squared_errors)
                gamma_mean = sum(variance_means) / len(variance_means)
                updates = []
                for steps in args.sampler_steps:
                    h = args.terminal_time / steps
                    lam = args.diffusion_scale * (1.0 - t_value)
                    update_mse = (h * lam / 2.0) ** 2 * error
                    brownian_mse = h * lam * gamma_mean
                    updates.append({
                        "steps": steps,
                        "step_size": h,
                        "lambda": lam,
                        "derivative_update_mse_estimate": update_mse,
                        "derivative_update_rmse_estimate": math.sqrt(update_mse),
                        "expected_brownian_increment_mean_square": brownian_mse,
                        "derivative_to_brownian_mse_ratio": update_mse / brownian_mse,
                    })
                record = {
                    "precision": precision, "t": t_value, "derivative_probes": count,
                    "independent_group_pairs": args.probe_repeats,
                    "actual_nfe": 2 * args.probe_repeats,
                    "actual_vjp": actual_vjp,
                    "derivative_mse_estimate": error,
                    "derivative_mse_estimate_per_pair": squared_errors,
                    "within_group_mse_estimate": (
                        sum(internal_errors) / len(internal_errors) if internal_errors else None
                    ),
                    "estimated_derivative_mean_square": sum(derivative_means) / len(derivative_means),
                    "mean_velocity_variance": gamma_mean,
                    "updates": updates,
                    **measurements,
                }
                results.append(record)
                print(json.dumps({"probe_case": record}, allow_nan=False), flush=True)
    precision_comparison = []
    for t_value in args.reference_times if {"fp32", "bf16"}.issubset(args.precisions) else ():
        fp32 = estimates_for_precision_comparison[("fp32", t_value)]
        bf16 = estimates_for_precision_comparison[("bf16", t_value)]
        precision_comparison.append({
            "t": t_value,
            "derivative_probes": max(args.probe_counts),
            "common_probe_fp32_bf16_rms_difference": float((fp32 - bf16).double().square().mean().sqrt()),
        })
    return results, precision_comparison


def sampler_benchmark(model, variant, config, labels, args, device):
    records, samples, comparisons = [], {}, []
    identities = {}
    for cutoff in args.cutoff_times:
        for precision in args.precisions:
            for steps in args.sampler_steps:
                for solver, count in [("ode", 1), *(("sde", m) for m in args.sampler_probes)]:
                    sampling = SamplingConfig(
                        solver=solver, steps=steps, terminal_time=cutoff,
                        terminal_readout="endpoint_mean",
                        diffusion_scale=args.diffusion_scale if solver == "sde" else 0.0,
                        diffusion_schedule="linear_decay", derivative_probes=count,
                        precision=precision, compile=args.compile,
                        allow_tf32=args.allow_tf32, cudnn_benchmark=args.cudnn_benchmark,
                    )
                    request = GenerationRequest(labels=labels.cpu(), sampling=sampling,
                                                batch_size=len(labels), seed=args.seed)
                    started = _cuda_measure(device)
                    with coupled_brownian_driver(steps, finest=max(args.sampler_steps)):
                        chunks = list(iter_loaded_generation(model, variant, config, request, device=device))
                    if len(chunks) != 1:
                        raise ValueError("bounded sampler benchmark must produce exactly one chunk")
                    measurements = _cuda_result(started, device)
                    result = chunks[0].sample_result
                    key = f"{precision}-{solver}-steps{steps}-probes{count}-t{cutoff}"
                    values = result.samples.detach().cpu()
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("sampler benchmark produced nonfinite samples")
                    samples[key] = values
                    identities[key] = (cutoff, precision, steps, solver, count)
                    record = {
                        "case": key, "sampling": sampling.to_dict(),
                        "actual_nfe": result.actual_nfe, "actual_vjp": result.actual_vjp,
                        "numerical_summary": result.numerical_summary,
                        "pixel_mean": float(values.double().mean()),
                        "pixel_mean_square": float(values.double().square().mean()),
                        "outside_display_range_fraction": float(values.abs().gt(1).float().mean()),
                        **measurements,
                    }
                    records.append(record)
                    print(json.dumps({"sampler_case": record}, allow_nan=False), flush=True)
                    del chunks, result
    for first, second in combinations(identities, 2):
        a, b = identities[first], identities[second]
        changed = [i for i in range(5) if a[i] != b[i]]
        if len(changed) != 1 or changed[0] == 3:
            continue
        kind = {
            0: "cutoff_common_normalized_random_inputs_not_same_physical_brownian_path",
            1: "precision_common_random_inputs",
            2: "step_refinement_coupled_brownian_path" if a[3] == "sde" else "ode_step_refinement",
            4: "probe_budget_common_brownian_path",
        }[changed[0]]
        comparisons.append({
            "kind": kind, "first": first, "second": second,
            "endpoint_pixel_rms_difference": float((samples[first] - samples[second]).double().square().mean().sqrt()),
        })
    return records, comparisons, samples


def run_sde_acceptance(args, protocol, *, config_path, job_context):
    validate_path_component(args.diagnostic_id, "diagnostic_id")
    job = job_context
    device = execution_device(job)
    if job.world_size != 1:
        raise ValueError("SDE acceptance runs in a single allocated GPU process")
    if device.type != "cuda":
        raise ValueError("SDE cost benchmark requires CUDA")
    if "bf16" in protocol.precisions and not torch.cuda.is_bf16_supported():
        raise ValueError("requested acceptance precision requires CUDA BF16 support")
    variant_root = require_canonical_variant_root(args.variant_root, job.runtime.experiments_root)
    bound = resolve_bound_epoch_checkpoint(variant_root, args.epoch)
    validate_bound_checkpoint_dataset(bound, job.runtime)
    model, variant, config = load_model_from_bound_checkpoint(
        bound, device, sampling=SamplingConfig(precision=protocol.precisions[0], compile=protocol.compile,
                                              allow_tf32=protocol.allow_tf32, cudnn_benchmark=protocol.cudnn_benchmark))
    requires_learned_variance(variant, feature="SDE acceptance")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _, heldout = make_datasets(config, job.runtime.dataset_root(config.dataset))
    if len(heldout) < protocol.batch_size:
        raise ValueError("held-out dataset is smaller than acceptance batch")
    # Seeded selection avoids taking only the first sorted ImageNet class.
    indices = torch.randperm(len(heldout), generator=posterior_generator(
        torch.device("cpu"), protocol.seed, "sde_acceptance", "reference_indices"
    ))[:protocol.batch_size].tolist()
    pairs = [heldout[index] for index in indices]
    x1 = torch.stack([pair[0] for pair in pairs]).to(device)
    labels = torch.tensor([pair[1] for pair in pairs], device=device, dtype=torch.long)
    x0 = config.noise_scale * torch.randn(
        x1.shape, generator=posterior_generator(torch.device("cpu"), protocol.seed, "sde_acceptance", "reference_noise"),
    ).to(device)
    parent = require_canonical_path(job.runtime.diagnostics_root, job.runtime.runs_root, name="diagnostics_root")
    staging, target = prepare_immutable_directory(parent, args.diagnostic_id)
    probes, precision = probe_benchmark(model, variant, config, x0, x1, labels, protocol, device)
    sampler_records, sampler_comparisons, samples = sampler_benchmark(model, variant, config, labels, protocol, device)
    sample_path = staging / "samples.pt"
    torch.save(samples, sample_path)
    with sample_path.open("rb") as handle:
        sample_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "schema": "pfm-sde-acceptance-v2",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_identity": runtime_identity(job),
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "checkpoint": {
            "path": str(bound.checkpoint_path), "sha256": bound.checkpoint_sha256,
            "epoch": args.epoch, "weights": "ema", "variant": variant,
            "dataset": config.dataset, "architecture": config.architecture,
            "noise_scale": config.noise_scale, "logvar_bounds": [config.logvar_min, config.logvar_max],
            "producer_runtime_identity": bound.checkpoint_runtime_identity,
        },
        "config": {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "protocol": {
            **asdict(protocol),
            "diffusion_schedule": "linear_decay", "terminal_readout": "endpoint_mean",
            "random_protocol": RANDOM_PROTOCOL,
            "brownian_comparison": "coarse normalized sums of finest-grid Gaussian increments at a shared cutoff",
            "derivative_comparison": "shared probe prefixes and aligned fine-grid indices",
            "reference_samples": "seeded held-out subset without replacement",
            "reference_indices": indices, "reference_labels": labels.cpu().tolist(),
            "probe_timing": "separate warmup excluded; sampler wall time includes execution setup and any recompilation",
        },
        "probe_benchmark": probes, "probe_precision_comparison": precision,
        "sampler_benchmark": sampler_records, "sampler_comparisons": sampler_comparisons,
        "samples": {"path": "samples.pt", "sha256": sample_digest},
        "scope": "bounded numerical stability and cost evidence; endpoint RMS is not FID or a proof of distributional convergence",
    }
    report_path = staging / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    json.loads(report_path.read_text())
    publish_immutable_directory(staging, target)
    print(json.dumps({"report": str(target / "report.json"), "checkpoint_sha256": bound.checkpoint_sha256}), flush=True)
    return 0
