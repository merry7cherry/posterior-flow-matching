# Posterior Flow Matching

Learn conditional velocity means and variances for generation and endpoint-posterior diagnostics.

**Chenrui Ma, Xi Xiao, Lin Zhao, Xinru Jiang, Tianyang Wang, Yanning Shen, Nando Fioretto**

[Paper (PDF)](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf) · [Project page](https://merry7cherry.github.io/posterior-flow-matching/) · [Running experiments](docs/reproduction.md) · [Architecture](docs/architecture.md)

The paper link opens the authors' preprint PDF hosted with the project page.
The arXiv submission is under review; its link will be added when available.

![Posterior Flow Matching](website/assets/method.webp)

PFM learns a Gaussian velocity posterior. Its mean drives ODE sampling; its
variance supports a corrected Itô SDE and endpoint uncertainty readouts. ODE
and SDE use the same trained model. Endpoint covariance is `(1-t)^2` times
velocity covariance; that scaling alone does not establish calibration.

This repository contains the **image implementation** (MNIST, CIFAR-10 and
ImageNet-256), including velocity/endpoint FM baselines, Full/Delta PFM,
training, sampling, and posterior diagnostics. Point-cloud, molecular and
PushT results on the project page belong to separate implementations. Their
code is not included in this image release. **We do not provide pretrained
checkpoints.** Train a model with the supplied interfaces, then use its saved
checkpoint for sampling and evaluation. Configurations are adjustable starting
points; their defaults are not exact recipes for the paper's reported results.

## Install

Python 3.13 is supported. Full image training targets Linux and NVIDIA CUDA;
CPU is available explicitly for small examples and tests. A CPU test does not
establish CUDA, compilation or multi-GPU support. See the current verification
record in [Running experiments](docs/reproduction.md#verification).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For FID/IS evaluation, install `python -m pip install -e '.[eval]'`. This uses
the pinned LTH14 `torch-fidelity` implementation used by the image protocol.
For CUDA 12.8 wheels, install PyTorch before the package:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,eval]'
```

Exact dependency locks are in `requirements/linux-cu128.lock` and
`requirements/macos.lock`. With `uv`, restore the Linux lock using
`uv pip sync --torch-backend cu128 requirements/linux-cu128.lock`, then
`uv pip install --no-deps -e .`.

## Prepare data

Data preparation is explicit; training never downloads datasets.

```bash
pfm-prepare-data --dataset mnist --data-root ./data --download
pfm-prepare-data --dataset cifar10 --data-root ./data --download
```

`--data-root` is the parent directory of `mnist/`, `cifar10/`, and
`imagenet1k-ilsvrc2012/`. For ImageNet, obtain the dataset separately and arrange
`data/imagenet1k-ilsvrc2012/{train,val}/<class>/...`, then run:

```bash
pfm-prepare-data --dataset imagenet --data-root ./data
```

Preparation checks the official splits and records their class/order identity.
It neither republishes ImageNet nor changes its labels. Existing prepared data
can be checked without `--download`.

## Train

A single process and standard `torchrun` share one training implementation:

```bash
pfm-train --config configs/cifar10_edm_ddpmpp.yaml --variant pfm_delta \
  --task-id cifar10 --data-root ./data --output-dir ./outputs --set epochs=600

torchrun --standalone --nproc_per_node=4 -m pfm.cli.train \
  --config configs/imagenet256_jit_b16_batch256.yaml --variant pfm_full \
  --task-id imagenet256 --data-root ./data --output-dir ./outputs --set epochs=600
```

Methods are `velocity_fm`, `endpoint_fm`, `pfm_full`, and `pfm_delta`.
Configuration is a complete YAML plus explicit `--set field=value` overrides.
Batch size is **per rank**; changing GPU count may change effective batch size
and scaled learning rate. See [Running experiments](docs/reproduction.md)
for configuration choices and evaluation requirements.

A small CPU workflow is provided in [examples/mnist_cpu.sh](examples/mnist_cpu.sh).
Its partial epochs test software execution, not image quality or convergence.
`limit_train_batches: null` runs full epochs; a positive integer (for example,
`--set limit_train_batches=2`) caps the batches per epoch and per rank.

Training writes `outputs/experiments/<task-id>/<method>/`, with immutable
`epochs/epoch_NNNN/checkpoint.pt` milestones and a rolling `recovery/latest.json`.
To continue, pass `--resume <variant-dir>/recovery/latest.json` with the same
training configuration and task. Only total epochs and checkpoint cadence may
change during exact resume. The batch limit must also remain the same. The
optimizer, EMA and rank RNG states are retained.

## Sample a checkpoint

A standalone checkpoint can be sampled without its original run directory or
training dataset. It always loads EMA weights. Use a new output directory:

```bash
pfm-sample --checkpoint ./checkpoints/checkpoint.pt \
  --config configs/sampling/cifar10_ode_visualization.yaml --output-dir ./samples-ode

pfm-sample --checkpoint ./checkpoints/checkpoint.pt \
  --config configs/sampling/cifar10_sde_visualization.yaml --output-dir ./samples-sde
```

Samples include endpoint-posterior variance maps and the effective protocol.
SDE requires a PFM checkpoint and a terminal cutoff below 1. The default
visualization captures uncertainty at `t=0.495` and renders final samples at
`T=0.99`; these are different times. Variance maps are not final continuation
variance or execution risk.

## Evaluate

Prepare the reference assets required by the selected metric, then evaluate
an explicit checkpoint. For example, CIFAR-10 FID uses the pinned LTH14 protocol:

```bash
pfm-prepare-fid --config configs/fid/cifar10_inception.yaml \
  --fid-asset-id cifar10-inception --data-root ./data --assets-root ./assets --download-weights

pfm-evaluate --checkpoint ./checkpoints/checkpoint.pt \
  --config configs/evaluation/cifar10_edm_ode.yaml \
  --set fid_asset_id=cifar10-inception \
  --data-root ./data --assets-root ./assets --output-dir ./evaluation-ode
```

Posterior diagnostics use `configs/evaluation/cifar10_edm_posterior.yaml` and
require the matching held-out dataset. FID, posterior diagnostics and bounded
sample grids have separate outputs. Standalone consumers retain `config.json`,
`checkpoint.json` and `result.json`; existing directories are never overwritten.
See [evaluation details](docs/evaluation.md) for scientific definitions and the
advanced multi-method aggregation workflow.

`--download-weights` explicitly prepares and checks the Inception weights.
All commands share the cache selected by `--cache-root` (default `.cache/pfm`).
Evaluation itself does not download missing weights or data.

## Development and citation

```bash
python -m pip install -e '.[dev,eval]'
python -m pytest -q
```

The full suite requires the optional evaluation backend. Tests cover posterior mathematics, samplers, models, data identity,
checkpoint/EMA/RNG recovery, and public entrypoints. CI runs a bounded CPU set.
GPU execution and paper-scale reproduction must be verified on their actual
hardware. Scheduler-specific deployment and historical migration tools are not
part of the public implementation.

Citation metadata is in [CITATION.cff](CITATION.cff). The paper is a preprint;
no conference acceptance is asserted. Third-party licensing is documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including the EDM-derived
backbone's CC BY-NC-SA 4.0 terms. Original PFM code and the project-page
implementation use the [MIT License](LICENSE); upstream components retain
their own licenses. Paper text, figures, datasets and model assets are not
relicensed by the code license.
