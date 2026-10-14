# Mainline architecture and configuration

`generator.py` combines the native PointNet encoder, 14-layer coupling-flow prior
(latent dimension 256 for official assets), and the pointwise PFM coordinate
network. All three are trained jointly. Coordinate hidden widths are
128/256/512/256/128. The velocity head uses a residual connection; a separate
context-conditioned log-variance head reads the last hidden layer. The sigmoid
mapping bounds coordinate log variance, with `[-3,3]` in the provided configs.

For a data endpoint x1, standard Gaussian x0, and t uniform inside
`[eps_t,1-eps_t]`, training constructs xt=(1-t)x0+t*x1. It minimizes the mean
coordinate Gaussian velocity NLL (dropping the fixed log(2*pi)/2 constant),
plus `kl_weight=0.001` times the native encoder/prior KL. The KL log density
uses the correct one-dimensional Gaussian normalizer before summing coordinates.
The latent condition path and loss accumulate in FP32.

The same checkpoint supplies two integrators. ODE uses Euler steps of the learned
mean velocity. SDE uses Euler–Maruyama with
`lambda(t) = diffusion_scale * (1-t)`, covariance `a = lambda(t)*Gamma_v`, and
drift `v + 0.5*lambda(t)*(div Gamma_v + Gamma_v*score)` where
`score=(t*v-x)/(1-t)`. The pointwise network permits an exact diagonal input
Jacobian with three VJPs per step; adding cross-point interactions to this network
would invalidate that derivative shortcut. SDE requires autograd for inputs even
when parameters are disabled, so use `no_grad`, not `inference_mode`.

Integration stops at `terminal_time=0.99`. One final forward call returns
`mu_x = x + (1-t)*v`; endpoint variance is `Gamma_x=(1-t)^2*Gamma_v`.
Trajectory snapshots include these endpoint means and variances. One hundred
integration steps therefore require 101 forward calls, plus the SDE's derivative
work. Forward-call counts do not represent matched runtime.

Chair and Airplane configs retain 2,048 points, batch size 128, seed 42, 800,000
updates, Adam LR 0.002, linear decay between updates 200,000 and 400,000 to
0.0001, clipping at 10, and no weight decay. CPU defaults use FP32. The original
research training used BF16 for the coordinate forward path; this release has
not validated that device-specific path. Evaluation uses 100 steps and SDE
`diffusion_scale=1.0` (the selected mainline setting), with seed 9988.
These settings do not constitute a claim that this release reproduces paper
metrics. `cpu_smoke.yaml` changes batch/schedule/step count for four updates;
it is explicitly a workflow check.

Normal resume loads one checkpoint containing the model, configuration,
optimizer, scheduler, iteration and RNG state. The saved subset limit and
iteration select the next native ordered batch. The checkpoint can be copied
and resumed in a different output directory, without the initialization asset.

Evaluation reports MMD/COV/1-NNA using squared Chamfer distance; the retained
algorithms are the native upstream definitions. Posterior diagnostics operate on
held-out reference interpolations with known endpoints at fixed times and a
separate random-time stream. They report residual second moments, posterior
variance, Gaussian NLL, bound occupancy and grouped variance/residual sums.
They are distinct from generated-state uncertainty readouts and do not establish
whole-shape confidence or calibrated free-generation quality.
