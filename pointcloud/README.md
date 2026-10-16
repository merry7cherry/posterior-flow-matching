# PFM — Point Clouds

Joint PFM for ShapeNet Chair and Airplane. The PointNet encoder, latent flow prior,
and coordinate network train together with Gaussian NLL and latent KL. A saved
model supports both ODE and posterior-driven SDE sampling.

[All tasks](../README.md) · [Project page](https://merry7cherry.github.io/posterior-flow-matching/) · [Paper](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf)

## Install

From `pointcloud/`, create a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]'
```

On macOS install `torch==2.8.0` from the normal package index instead. Commands
default to CPU with two threads. No CUDA extension is needed.

## Data

Obtain `shapenet.hdf5` and the category's `GEN_chair.pt` or `GEN_airplane.pt`
from the [original assets](https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ).
[Data details](docs/data.md) describes the split and normalization. The upstream
asset initializes the encoder and latent prior as ordinary trainable parameters;
its diffusion decoder is unused. Data and pretrained PFM weights are not bundled.

## Quick start

This uses four real shapes, 2,048 points and the native network, for four updates:

```bash
pfm-pointcloud-train --config configs/cpu_smoke.yaml \
  --dataset /path/to/shapenet.hdf5 --condition /path/to/GEN_chair.pt \
  --output runs/smoke --limit-train 4
pfm-pointcloud-sample --checkpoint runs/smoke/final.pt \
  --output runs/smoke/samples --count 2 --steps 2
```

## Train and resume

```bash
pfm-pointcloud-train --config configs/chair.yaml \
  --dataset /path/to/shapenet.hdf5 --condition /path/to/GEN_chair.pt \
  --output runs/chair

pfm-pointcloud-train --resume /path/to/latest.pt \
  --dataset /path/to/shapenet.hdf5 --output runs/continued
```

For Airplane select `configs/airplane.yaml` and its initialization asset.
Checkpoints contain model, configuration, optimizer, scheduler, iteration and RNG
state. `--stop-after 2` stops after two updates while retaining the configured
schedule. PointNet BatchNorm needs at least two shapes
per batch, including the last batch. Local losses are written to `train.jsonl`.

## ODE/SDE sampling

```bash
pfm-pointcloud-sample --checkpoint runs/chair/final.pt --output samples/ode --solver ode
pfm-pointcloud-sample --checkpoint runs/chair/final.pt --output samples/sde --solver sde
```

Sampling uses the checkpoint's latent prior, without a dataset. It writes
`points.npy`, a PNG gallery and endpoint-posterior trajectory snapshots. The
checkpoint configuration provides steps, terminal time and linear SDE scale;
`--steps` and `--diffusion-scale` override these explicitly.

## Generation metrics

```bash
pfm-pointcloud-evaluate --checkpoint runs/chair/final.pt \
  --dataset /path/to/shapenet.hdf5 --output evaluation/chair
```

Evaluation uses matched initial draws for ODE/SDE, shape bounding-box
normalization, and native squared-Chamfer MMD, COV and 1-NNA. MMD averages each
reference shape's nearest generated distance; 1-NNA is best near 50%. For a small
run use `--limit 2 --steps 2`.

## Posterior diagnostics

```bash
pfm-pointcloud-diagnostics --checkpoint runs/chair/final.pt \
  --dataset /path/to/shapenet.hdf5 --output diagnostics/chair.json
```

This directly evaluates held-out reference interpolations, without first running
generation metrics. It reports raw residual/variance moments and Gaussian NLL in
velocity and endpoint coordinates. Use `--limit 2` to select two reference shapes.

## Code guide

- `pfm_pointcloud/pfm.py`: bounded variance, NLL, exact pointwise divergence, ODE/SDE.
- `generator.py`, `encoder.py`, `flow.py`: joint encoder and latent prior.
- `train.py`: interpolation, joint loss, optimization and ordinary checkpoints.
- `generate.py`, `evaluate.py`, `diagnostics.py`: direct task entrypoints.

[Method details](docs/method.md)

## Citation and license

Use the [PFM citation](../CITATION.cff). Also cite Luo and Hu,
*Diffusion Probabilistic Models for 3D Point Cloud Generation*, CVPR 2021,
when using the backbone and data pipeline. Code is MIT; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External data retain their own terms.
