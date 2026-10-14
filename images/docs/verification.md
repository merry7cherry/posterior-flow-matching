# CPU verification · October 13, 2026

Local checks used macOS arm64, Python 3.13.12 and PyTorch 2.11.0, with two CPU
threads. The source suite and the independently installed wheel each passed
12 tests. Tests cover both mean parameterizations, the Gaussian NLL/score,
variance input derivatives, the three native architecture implementations,
zero-noise ODE/SDE equivalence, batch-independent random draws, EMA/checkpoint
continuation, raw-moment/NLL diagnostics and streaming metric statistics.

A fixed-input comparison against the prior public image source at
`3b00c5af9caa01057a03ae35c01ec5dd294d1702` used reduced U-Net, DDPM++ and JiT models
with both velocity and endpoint parameterizations. Only tensor-key names were
mapped between the old heads and the unified mean/log-variance heads. Outputs,
NLL, all parameter gradients, two-step ODE/SDE samples, captured endpoint
variance, and indexed initialization matched exactly in all six cases. The SDE
comparison used positive diffusion and two derivative probes. This checks the
preserved computations; it is not full-model training or convergence evidence.

The wheel was built and installed with its dependencies in a fresh Python
environment. Its documented MNIST example used the real training/test data:
two optimizer steps, save, two more after resume, four ODE images, four SDE
images, and diagnostics on eight held-out images at three reference times.
The final checkpoint recorded epoch 2 and optimizer step 4. A synthetic
continuous-versus-resumed training test matched model and EMA tensors exactly.
All four installed commands resolved; installed-checkpoint sampling also ran
outside the source directory without a Git checkout.

The pinned LTH14 Inception backend was additionally exercised on 20 synthetic
RGB images. Streaming FID and IS returned finite results; the same-population
FID was approximately -0.000046, a small matrix-square-root roundoff residual
around zero for this low-rank smoke population. This is a metric execution
check, not an image-quality result. Reference statistics and generated-image
quantization retain the native metric definitions.

The source distribution includes configurations, documentation and the CPU
example. The wheel includes the applicable license texts. Temporary checkpoints,
images, metrics, parity scripts and logs remain outside this repository.

These local checks do not establish CUDA, compilation, distributed training,
full-scale FID/IS, or the paper's generation quality. The training/sampling
settings are documented in the task README and remain explicit YAML options.
