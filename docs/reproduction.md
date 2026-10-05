# Running experiments

This guide covers the image implementation's training, sampling and evaluation
interfaces. We do not provide pretrained checkpoints. Train your own model and
use its saved checkpoint with the consumers below. The supplied configurations
are adjustable starting points, not exact recipes for the paper's reported
numbers. A small smoke run verifies software behavior, not generation quality.

## Install and prepare data

Follow the [installation and data preparation instructions](../README.md#install).
Training defaults to CUDA; CPU must be selected explicitly with `--device cpu`
and `--set mixed_precision=none`. Data preparation is a separate command and
requires explicit download permission through `--download`.

The package implements MNIST, CIFAR-10 and ImageNet-256. Point-cloud, molecular
and PushT results on the project page belong to separate implementations.

## Small executable example

From the repository root, with the package installed, run:

```bash
bash examples/mnist_cpu.sh /path/to/a/new/smoke-directory
```

The example explicitly downloads MNIST, trains two batches, resumes for two
more, samples with ODE and positive-diffusion SDE, and evaluates an eight-image
posterior population. The two epochs are partial epochs. The tiny model has no
scientific-quality claim. Effective configurations, checkpoints, grids and
numerical diagnostics remain in the selected directory.

`limit_train_batches: null` runs a full epoch; a positive integer caps batches
per epoch and per rank. The example selects `--set limit_train_batches=2`.
This field is recorded in the effective configuration, training protocol and
checkpoint, and must remain the same on resume.

## Choose a configuration and train

A complete YAML plus typed `--set field=value` overrides defines a run.
Batch size is per rank. Choose the data/output paths and device explicitly;
changes to batch size, world size, precision or workers change the training
contract. Configuration files cover these starting points:

| Purpose | Training configuration | Method | Consumer configuration |
| --- | --- | --- | --- |
| MNIST small example | `configs/mnist_unet.yaml` with small overrides | `pfm_delta` | `configs/sampling/mnist_visualization.yaml`, `configs/evaluation/mnist_posterior.yaml` |
| CIFAR-10 | `configs/cifar10_edm_ddpmpp.yaml` | `pfm_delta` | `configs/evaluation/cifar10_edm_ode.yaml`, `configs/evaluation/cifar10_edm_sde.yaml` |
| CIFAR-10 variance-bound variants | `configs/cifar10_edm_ddpmpp_logvar_neg2.yaml`, `configs/cifar10_edm_ddpmpp_logvar_neg3.yaml` | PFM | Same explicit ODE/SDE and posterior consumers |
| ImageNet-256 JiT-B/16 | `configs/imagenet256_jit_b16.yaml` or `configs/imagenet256_jit_b16_batch256.yaml` | `pfm_full` | `configs/evaluation/imagenet256_jit_ode.yaml`, `configs/evaluation/imagenet256_jit_sde.yaml` |
| Posterior diagnostics | Matching checkpoint's training configuration | PFM only | `configs/evaluation/*_posterior.yaml` |
| FM comparisons | Same dataset and training configuration | `velocity_fm`, `endpoint_fm` | ODE consumers |

The two ImageNet configurations use different per-rank batch sizes and clipping
settings. Select a configuration as a whole, and record overrides with the run.
The [README training examples](../README.md#train) cover a single CUDA process
and standard multi-GPU `torchrun` using the same implementation.

Training writes immutable `epochs/epoch_NNNN/checkpoint.pt` milestones and a
rolling `recovery/latest.json` under the task/method directory. To continue,
repeat the training command with the same configuration, paths and task ID,
and add `--resume <variant-dir>/recovery/latest.json`. Only total epochs and
checkpoint cadence may change. Model, EMA, optimizer, rank RNG, data identity,
world size and the batch limit are retained and checked.

## Sample and evaluate your checkpoint

Copy your selected `epochs/epoch_NNNN/checkpoint.pt` to any directory. File
consumers need no sibling manifest or original experiment tree and load EMA
parameters. Use a fresh output directory for each result:

```bash
pfm-sample --checkpoint /path/to/checkpoint.pt \
  --config configs/sampling/cifar10_ode_visualization.yaml \
  --output-dir ./samples-ode

pfm-sample --checkpoint /path/to/checkpoint.pt \
  --config configs/sampling/cifar10_sde_visualization.yaml \
  --output-dir ./samples-sde
```

These examples use a CIFAR-10 checkpoint. Select the matching configuration for
other datasets. SDE requires a PFM model and a terminal cutoff below one. Change
sampling settings explicitly, for example `--set sampling.steps=100`; no default
sampler configuration promises a particular FID or another paper result.

Samples record their effective configuration, checkpoint digest and producer
metadata. Posterior evaluation additionally checks the prepared dataset:

```bash
pfm-evaluate --checkpoint /path/to/checkpoint.pt \
  --config configs/evaluation/cifar10_edm_posterior.yaml \
  --data-root ./data --output-dir ./posterior-evaluation
```

CPU sampling uses `--device cpu --set sampling.precision=fp32` and requires an
uncompiled consumer (`--set sampling.compile=false` when needed). CPU posterior
evaluation uses `--device cpu --set precision=fp32 --set compile=false`.
The executable MNIST example supplies bounded CPU settings for both paths.

The CLI also supports `--variant-root` and `--epoch` for consumers attached to
a native experiment directory, with explicit `--sample-id`/`--evaluation-id`.
Those bound results can be passed to `pfm-aggregate`; standalone file results
are not aggregate inputs.

## Prepare metric assets

Install the `eval` extra for FID/IS. Use `pfm-prepare-fid` with the matching
`configs/fid/` profile and supply its asset ID in the evaluation configuration;
see the [complete FID example](../README.md#evaluate). On the first preparation,
`--download-weights` explicitly downloads and verifies the third-party
Inception feature extractor. These metric weights are separate from PFM models.
Preparation and evaluation must use the same `--cache-root`; evaluation reads
prepared assets and never downloads weights or datasets.

MNIST conditional accuracy requires a separately trained classifier asset.
For explicit CPU FP32 training:

```bash
pfm-train-classifier --config configs/classifier/mnist.yaml \
  --classifier-asset-id mnist-classifier --data-root ./data \
  --output-dir ./outputs --assets-root ./assets --device cpu \
  --set mixed_precision=none
```

This is classifier training, not the bounded two-batch PFM smoke example.
A BF16 classifier configuration with `--device cpu` is rejected before training;
the command must declare its actual precision.

For matched comparisons, hold the evaluation population, class balance,
preprocessing, seed, terminal readout, steps, diffusion schedule and derivative
probes fixed. Report model forwards, VJPs, elapsed time and memory separately.
See [Evaluation](evaluation.md) for metric definitions and recorded protocols.

## Verification

The 2026-10-05 working-tree candidate was checked on macOS arm64, Python
3.13.12 and PyTorch 2.11.0 CPU:

- All 1,114 collected tests passed across the suite and targeted reruns. The
  sandboxed suite passed 1,110 tests; three DataLoader worker tests required a
  rerun with local shared-memory access, and one real Inception backend check
  passed separately using already verified cached weights.
- The rebuilt wheel installed in a separate environment. Its 93 Python source
  files matched the working tree byte for byte, and all seven console commands
  resolved. Thirteen documented commands parsed with valid configurations.
- The installed wheel completed the MNIST example: two training batches, two
  more after resume, four ODE samples, four SDE samples and an eight-image
  posterior evaluation. Both checkpoints recorded `limit_train_batches=2`.
- Classification tests exercised CPU BF16 rejection and real small FP32
  training with matching saved precision metadata.

Logs, configuration/checkpoint read-backs and browser checks are retained with
the local outputs in `outputs/release-review-20261005/`; they are excluded from
the release snapshot. These checks cover a modified working tree, not an exact
clean Git revision or converged model quality.

The default GitHub workflow runs bounded CPU numerical/model checks. CUDA,
compiled input derivatives and NCCL multi-GPU execution require evidence from
the corresponding hardware. CPU tests and configuration files do not establish
GPU acceptance or paper-scale generation quality.

## Run identity and paper assets

A checkout with clean runtime sources records its Git SHA. Changes to `src/`,
`configs/`, `pyproject.toml`, dependency locks or `.python-version`, and
installations without Git metadata, record `null`. Unknown revision must not be
interpreted as an exact published source identity. Historical checkpoints retain
their original producers; this release does not migrate deployment metadata.

Website figures and tables come from the preprint sources listed in
`website/assets/provenance.json`. They report the paper's experiments, not
outputs regenerated by the small executable example. The paper link points directly to the preprint PDF in the project page
assets. An arXiv link will be added when available. Original PFM code uses MIT;
third-party terms are preserved in the repository license notices.
