# PFM · Images

[Project](https://merry7cherry.github.io/posterior-flow-matching/) · [Paper](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf) · [All four tasks](../README.md)

PFM image generation with DDPM++ on CIFAR-10 and JiT-B/16 on ImageNet-256.
A small conditional U-Net on MNIST provides the CPU example. Each model predicts
a mean and bounded velocity log-variance; the same learned posterior supplies
ODE/SDE sampling and endpoint uncertainty readouts. Pretrained PFM weights are
not included.

## Install

Run these commands from the monorepo's `images/` directory, in an environment
separate from the other three tasks. Python 3.13 is supported.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For FID/IS, add `python -m pip install -e '.[eval]'`. The evaluation extra pins
the LTH14 `torch-fidelity` implementation used by the image protocol.
For CUDA, install the appropriate PyTorch 2.11.0 / torchvision 0.26.0 wheels
before installing this package.

## Quick start

```bash
bash examples/mnist_cpu.sh ./outputs/mnist
```

This downloads MNIST, trains two batches, resumes for two more, generates ODE
and SDE samples, and computes posterior diagnostics on eight held-out images.
Pass an existing MNIST directory as the second argument to reuse local data.

## Data and training

`--data-root` points directly to the dataset directory. MNIST and CIFAR-10 can
be prepared with the explicit download command:

```bash
pfm-images-prepare data --dataset cifar10 --data-root ./data/cifar10
pfm-images-train --config configs/cifar10.yaml --data-root ./data/cifar10 \
  --output ./outputs/cifar10 --device cuda
```

Obtain ImageNet separately and organize it as
`./data/imagenet/{train,val}/<class>/...`, with the same class names in both
splits. Train with the native JiT-B/16 configuration:

```bash
torchrun --standalone --nproc_per_node=4 -m pfm.train \
  --config configs/imagenet256.yaml --data-root ./data/imagenet \
  --output ./outputs/imagenet --device cuda
```

Batch size is per process. The ImageNet configuration scales its base learning
rate by global batch size / 256; four processes give batch 1024 and learning
rate 0.0002. Training uses a five-epoch warmup followed by a constant rate.
CIFAR-10 uses its explicit fixed learning rate. YAML values can be overridden
with `--set field=value`. Full epochs use `limit_train_batches: null`.

`mean_parameterization: velocity` is used for CIFAR-10; `endpoint` is used for
ImageNet, as in the paper's endpoint-mean appendix. Both minimize the same PFM
velocity-space NLL and predict velocity variance. They are two parameterizations
of PFM, not separate methods.

Training writes `latest.pt`, periodic `epoch_NNNN.pt`, `config.yaml`, and
`loss.jsonl` directly under `--output`. Checkpoints contain model, EMA, optimizer,
configuration, epoch/step and PyTorch RNG states. Continue from an epoch boundary:

```bash
pfm-images-train --resume ./outputs/cifar10/latest.pt --data-root ./data/cifar10 \
  --output ./outputs/cifar10 --device cuda --set epochs=1000
```

Continuation keeps the saved model/data settings and process count.

## ODE and SDE sampling

```bash
pfm-images-sample --checkpoint ./outputs/cifar10/latest.pt \
  --sampling configs/ode.yaml --output ./outputs/ode --device cuda
pfm-images-sample --checkpoint ./outputs/cifar10/latest.pt \
  --sampling configs/sde.yaml --output ./outputs/sde --device cuda
```

Sampling loads EMA weights and saves `samples.pt`, an image grid, and an endpoint
variance grid. The saved tensors include velocity variance, endpoint variance
and their capture time. By default the readout is captured halfway along the
integration grid (`t=0.495` for these presets), while generated images use the
endpoint mean at `T=0.99`.

The SDE uses `lambda(t) = diffusion_scale * (1-t)` in both its corrected drift
and noise. CIFAR-10's preset uses the paper's coefficient 0.5 and 50 steps.
For ImageNet use `--set steps=250`; its SDE starting setting additionally uses
`--set diffusion_scale=0.1`. The coefficient remains configurable.
Velocity variance is bounded by the model configuration; endpoint variance is
`(1-t)^2` times velocity variance and describes the endpoint posterior at the
readout state.

## Evaluation

Prepare FID statistics from the original CIFAR-10 training split:

```bash
pfm-images-prepare fid --config configs/cifar10.yaml --data-root ./data/cifar10 \
  --output ./data/cifar10_train_fid.npz --device cuda
pfm-images-evaluate generation --checkpoint ./outputs/cifar10/latest.pt \
  --sampling configs/ode.yaml --fid-stats ./data/cifar10_train_fid.npz \
  --count 50000 --output ./outputs/cifar10_fid.json --device cuda
```

For ImageNet use `configs/imagenet256.yaml`, its training split, 250 sampling
steps, and `--inception-score` to report IS alongside FID. Sampling cycles class
labels, yielding balanced populations when the count is divisible by the number
of classes. Generated images use the native clamp/round-to-uint8 preprocessing;
ImageNet reference images use the native ADM/JiT center crop. The Inception
weights are downloaded by the metric backend on first use, or supplied through
`--inception-weights /path/to/weights.pth` to preparation and evaluation.

Held-out posterior checks need no Inception model:

```bash
pfm-images-evaluate posterior --checkpoint ./outputs/cifar10/latest.pt \
  --data-root ./data/cifar10 --output ./outputs/posterior.json --device cuda
```

These evaluate independent noise/data pairs at fixed reference times and report
raw residual means, squared residuals versus predicted variance, variance bins,
and Gaussian NLL in velocity/endpoint coordinates. Use `--times`, `--bins`, and
`--count` to select the diagnostic population. MNIST supports posterior diagnostics.

## Code guide

The main path is `train.py` → `probability.py` → `sampling/samplers.py` →
`evaluate.py`. Native architectures live in `models/`; `diagnostics.py` and
`metrics.py` contain the task measurements. There is one sampler shared by
sample grids and generation metrics. See [architecture](docs/architecture.md).

Run the tests with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

Original PFM code uses MIT. The EDM-derived backbone retains CC BY-NC-SA 4.0;
JiT retains MIT attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Paper figures, datasets and model weights retain their own terms.
