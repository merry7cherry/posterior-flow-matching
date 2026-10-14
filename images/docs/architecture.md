# Image implementation

`train.py` loads a dataset and a native model, constructs
`xt = (1-t)*x0 + t*x1`, minimizes the PFM Gaussian NLL, updates EMA and saves an
ordinary checkpoint. `checkpoint.py` loads that EMA model directly for inference.

`models/` retains the conditional U-Net, DDPM++ and JiT-B/16 computations.
All three use the same mean/log-variance heads. `probability.py` interprets the
mean as velocity or endpoint coordinates and expresses the loss, score and SDE
coefficients in velocity coordinates. Sensitive posterior computations are FP32.

`sampling/samplers.py` contains both Euler ODE and corrected Itô Euler--Maruyama.
The latter differentiates the effective variance with respect to the current
state using Rademacher VJPs. Its drift is `v + lambda/2*(div(gamma)+gamma*score)`
and noise covariance is `h*lambda*gamma`. `generate.py` batches this one sampler
for grids and generation metrics. Fixed-index noise/probe streams retain the
native random draws when batch partitions change; they do not identify artifacts.

`metrics.py` computes streaming FID/IS from the pinned LTH14 extractor.
`diagnostics.py` evaluates raw moments and Gaussian NLL on held-out reference
interpolations. Endpoint means/variances are derived by the affine mapping.
`sample.py` writes one simple image grid and its endpoint-variance grid.

Each public operation takes ordinary input/output paths. A training checkpoint
stores the tensors and settings needed for continuation. Sampling and evaluation
use the same checkpoint loader, without requiring its original directory.
