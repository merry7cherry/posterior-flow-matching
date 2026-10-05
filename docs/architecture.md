# Architecture

PFM has one scientific execution path:

```text
explicitly prepared dataset -> training -> explicit checkpoint
                                         -> ODE/SDE samples and variance maps
                                         -> generation metrics
                                         -> posterior diagnostics
```

The public runtime uses user-selected paths and Python/torchrun. It has no
Slurm, cluster-account, deployment-lock or historical migration service.

## Ownership

| Module | Responsibility |
| --- | --- |
| `core` | Validated configurations, method identities, execution paths/process identity, checkpoint contracts |
| `data` | Explicit preparation, official split loading, class/order identity |
| `models` | Backbones, prediction heads and outputs |
| `probability` | Gaussian objectives' readouts, velocity/endpoint moments, score and SDE coefficient algebra |
| `training` | Optimization, DDP, EMA, exact resume and checkpoint publication |
| `sampling` | Time grids, random streams, variance derivatives and ODE/SDE updates |
| `diagnostics` | Posterior mathematics and bounded numerical accumulators |
| `visualization` | Figures derived from numerical results |
| `artifacts` | Atomic, non-overwriting output publication and result bindings |
| `evaluation` | Checkpoint loading, shared generation, metrics and aggregation |
| `cli` | One request into these services; no scheduler orchestration |

`probability.targets` owns interpolant-time broadcasting. `variant_metadata`
owns the four supported methods. `SamplingConfig` owns the sampling protocol;
training backend choices never silently override inference settings.

## Execution and configuration

`core.execution.RuntimePaths` resolves explicit project/data/output/assets/cache
paths. `ExecutionContext` records device and standard torchrun rank identity.
It records a source SHA only for a clean Git checkout; installed or dirty source
has unknown (`null`) revision. Device selection is explicit and CUDA never
silently falls back to CPU. Multi-GPU training uses NCCL.

A full YAML plus typed `--set` overrides is normalized by its owning schema.
Tracked-run commands freeze canonical configuration bytes under the output
workspace; independent-checkpoint consumers record their full effective
configuration beside the result. Input formatting is not scientific identity.
Dataset preparation is a separate explicit action, never a training side effect.

`limit_train_batches` is part of the training configuration: `null` runs a full
epoch, while a positive integer caps batches per epoch and per rank. It is
serialized with checkpoints, included in the training protocol identity, and
must stay fixed on resume. Small examples set it through the same `--set`
interface as other training choices.

## Methods and numerics

Velocity FM, endpoint FM, Full PFM and Delta PFM share the corresponding
backbone and training protocol. PFM adds bounded velocity log-variance.

Posterior means/variances and sensitive math remain FP32 even with BF16 model
execution. Endpoint covariance is `(1-t)^2` times velocity covariance. It is
not final rollout risk, and its decay alone is not calibration evidence.

ODE/SDE use one sampler and one shared posterior readout. Positive-diffusion
SDE uses a cutoff below 1 and the full drift correction, with independent
Rademacher VJP probes. Zero diffusion uses the corresponding ODE update and
readout without Brownian draws or variance derivatives. Finite-step/probe
approximations are not exact dynamics.

## Checkpoints and outputs

Training writes a task/method directory under `outputs/experiments`. Atomic
rolling recovery is separate from immutable milestone checkpoints. Model,
EMA, optimizer, training configuration, completed progress, world size,
rank-complete RNG and dataset identity are retained. A single writer owns a
training directory; existing immutable outputs cannot be overwritten.

Exact resume checks configuration, data identity, source revision when
recorded, and world size. Only total epochs and checkpoint cadence may change.
There is no historical-state migration or arbitrary warm-start bypass.

A standalone checkpoint file can be sampled without its original directory
or dataset. Evaluation checks the prepared dataset and selected metric assets.
Both file consumers use the same model loading, generator and metric services
as experiment-bound consumers; neither implements a second sampler.

Sample grids are qualitative outputs. Generation metrics and posterior
metrics remain separate. The advanced aggregate command takes an explicit
set of bound numerical results; it does not discover results recursively or
load checkpoints. See [evaluation.md](evaluation.md) for its detailed contract.

## Verification

Use unit/numerical checks for the scientific invariants and end-to-end tests
for training, recovery and consumers. CPU evidence does not establish CUDA,
compiled kernels or DDP. Publish the tested revision and actual environment
with any hardware-specific result. Public release readiness and paper-scale
scientific reproduction are separate checks.
