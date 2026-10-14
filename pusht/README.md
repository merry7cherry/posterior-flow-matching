# PFM — PushT

Image-conditioned PushT with a trainable native visual encoder and an action U-Net.
Both heads learn the Gaussian velocity posterior through one joint NLL objective;
the same checkpoint supplies ODE and posterior-driven SDE action plans.

[Project](https://merry7cherry.github.io/posterior-flow-matching/) ·
[Paper](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf) ·
[Repository](https://github.com/merry7cherry/posterior-flow-matching) ·
[Images](../images/) · [Point clouds](../pointcloud/) · [Molecules](../molecules/)

## Install

Run these commands from `pusht/`, using Python 3.10 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.10
# Linux CPU wheels; skip this line on macOS.
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --excludes requirements-exclude.txt -e '.[test]'
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy OMP_NUM_THREADS=2
pytest -q
```

The pinned Gym/Pymunk versions preserve the native simulator. The exclusion removes
robomimic's unused EGL discovery dependency. Each PFM task uses its own environment.
The commands also work from a built wheel; no Git metadata is needed at runtime.

## Data

Download the [native PushT demonstrations](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip):

```bash
mkdir -p data
curl -fL https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -o data/pusht.zip
python -m zipfile -e data/pusht.zip data
```

Use the resulting `data/pusht/pusht_cchi_v7_replay.zarr`: 206 episodes and 25,650
frames. The native seeded split uses 90 training episodes and the complementary
validation episodes. The upstream normalization is stored in the checkpoint.
Data and trained weights are not bundled; dataset terms remain with its publisher.
Use absolute data paths when running commands outside this directory.

## CPU example

The smoke configuration keeps the native visual encoder, action interface and
simulator, with a narrower U-Net and two single-update epochs:

```bash
pfm-pusht-train --config-name=smoke \
  task.dataset.zarr_path=/absolute/path/pusht_cchi_v7_replay.zarr \
  hydra.run.dir=outputs/smoke
pfm-pusht-evaluate --checkpoint outputs/smoke/checkpoints/final.ckpt \
  --output outputs/smoke/ode --solver ode --steps 2 --n-test 1 --max-steps 8
pfm-pusht-evaluate --checkpoint outputs/smoke/checkpoints/final.ckpt \
  --output outputs/smoke/sde --solver sde --steps 2 --n-test 1 --max-steps 8
pfm-pusht-diagnose --checkpoint outputs/smoke/checkpoints/final.ckpt \
  --data /absolute/path/pusht_cchi_v7_replay.zarr --output outputs/smoke/posterior
```

This checks execution on CPU, not task quality or paper metrics.

## Train and resume

```bash
pfm-pusht-train task.dataset.zarr_path=/absolute/path/pusht_cchi_v7_replay.zarr \
  hydra.run.dir=outputs/pfm
```

`pusht.yaml` sets the full U-Net widths 512/1024/2048, batch64, AdamW LR1e-4,
500 warmup updates, cosine decay, 3050 epochs and per-update EMA. CPU/FP32 is
the default; full training is expensive.

Resume from an ordinary checkpoint, including in a different output directory:

```bash
pfm-pusht-train task.dataset.zarr_path=/absolute/path/pusht_cchi_v7_replay.zarr \
  training.resume=/absolute/path/outputs/pfm/checkpoints/latest.ckpt \
  hydra.run.dir=outputs/resumed
```

Keep the intended model, data and training configuration when resuming. Checkpoints
store model/EMA weights, optimizer, scheduler, epoch/update counters and RNG state.
Recovery starts after the saved completed epoch. `latest.ckpt` and `final.ckpt`
are written under `checkpoints/`; scalar training and evaluation logs append to
`logs.json.txt`, including when resuming in the same output directory. Load only
trusted PyTorch files.

## Sample and evaluate

```bash
pfm-pusht-evaluate --checkpoint outputs/pfm/checkpoints/final.ckpt \
  --output outputs/ode20 --solver ode --steps 20
pfm-pusht-evaluate --checkpoint outputs/pfm/checkpoints/final.ckpt \
  --output outputs/sde50 --solver sde --steps 50 --diffusion-scale 0.1
```

Each plan conditions on two observed frames, predicts 16 actions, executes
`action_pred[:,1:9]`, and replans. Evaluation uses EMA when enabled, 50 simulator
starts, and at most 300 environment steps. `--n-envs` controls parallelism.
`evaluation.json` reports mean per-episode maximum normalized block overlap;
`media/` contains a rollout video. Sampling is part of this closed-loop evaluation.

## Inspect the posterior

```bash
pfm-pusht-diagnose --checkpoint outputs/pfm/checkpoints/final.ckpt \
  --data /absolute/path/pusht_cchi_v7_replay.zarr --output outputs/posterior \
  --batches 2
```

This command runs independently of the simulator. `diagnostics.json` reports
coordinate velocity variance, squared velocity residual and Gaussian NLL without
its additive constant on held-out reference interpolations at t=0.1,0.5,0.9,0.99.
Endpoint variance is `(1-t)^2 Gamma_v` in normalized action coordinates.
These readouts describe action uncertainty, not object-position confidence.

## Code and method

- `policy/pfm_image_policy.py`: shared observation encoding, NLL and action plans.
- `model/pfm.py`: bounded log variance, covariance derivatives, Euler ODE and
  corrected Itô Euler–Maruyama SDE with `lambda(t)=lambda0*(1-t)`.
- `workspace/train_pfm_workspace.py`: training, EMA and checkpoint/resume.
- `diagnostics.py`: reference posterior checks; `env_runner/`: native scoring.

These paths are under `diffusion_policy/`, the retained upstream namespace.
The encoder uses crop84, ResNet and GroupNorm. Log-variance bounds are `[-3,3]`.
Four Rademacher VJP probes estimate the diagonal variance derivative; the SDE
includes both divergence and covariance-weighted score corrections. Both samplers
stop at t=0.99 and use the endpoint mean readout.

[VERIFICATION.md](VERIFICATION.md) records CPU evidence. Full training convergence,
GPU behavior, timing and paper quality metrics have not been established by these
checks. Cite PFM via [CITATION.cff](CITATION.cff), and
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for its native
architecture, data pipeline and simulator. See [third-party notices](THIRD_PARTY_NOTICES.md)
and [LICENSE](LICENSE) for the retained MIT/Apache-2.0 terms.
