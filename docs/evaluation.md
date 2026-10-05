# Evaluation Contract

Every evaluation product consumes one explicit immutable checkpoint. The
project separates common quality, PFM posterior analysis, and qualitative
visualization so each profile has a clear population and artifact policy.

## Generated common quality

`generated_common` is method-neutral and streams metrics without retaining
generated images, variance maps, grids, or tensor chunks.

| Dataset | Metrics | Generated population | Default ODE steps |
| --- | --- | ---: | ---: |
| MNIST | Conditional accuracy | 10,000, class-balanced | 50 |
| CIFAR-10 | FID | 50,000, class-balanced | 50 |
| ImageNet-256 | FID and Inception Score | 50,000, 50 per class | 250 |

Canonical production generation uses seed `321` and EMA weights. CIFAR-10 FID
uses all 50,000 official training images and 50,000 balanced generated images
under the pinned LTH14 torch-fidelity protocol. The tracked asset binds the
backend revision, full resolved weights digest, preprocessing, dataset identity,
and actual counts.

The shared ODE/SDE output remains the raw floating-point generated stream. MNIST
conditional accuracy evaluates a non-mutating
`raw_generated_sample.clamp(-1, 1)` view; it performs no uint8 conversion or
renormalization. FID independently applies its pinned clamp-and-quantize
protocol to the unchanged raw stream. Generated results record this classifier
input protocol, so conditional accuracy produced under another consumer
protocol is not treated as directly comparable.

## Posterior diagnostics

Posterior analysis is PFM-only and keeps two independently reproducible suites:

- `controlled`: fixed-time analysis over the full official evaluation split;
- `random_time`: an independent full-split calibration pass using the same
  dataset-level time distribution as training.

The interpolant direction is
`x_t = (1-t)x_0 + t x_1`, with `t=0` at noise and `t->1` at data. The model
predicts velocity-space diagonal variance:

```text
Gamma_v = exp(s)
Gamma_x = (1-t)^2 Gamma_v
endpoint_nll = velocity_nll + log(1-t)
```

Both velocity and endpoint views retain component-average predicted variance
and squared error, error-to-variance ratio, complete NLL, absolute/relative
calibration error, tie-correct Spearman correlation, calibration bins,
histograms, and time-calibration summaries.

The ratio is `sum(squared_error) / sum(predicted_variance)`, a
variance-weighted average of standardized squared residuals. It is retained
alongside the distinct unweighted standardized residual moments. Both suites
report shared `standardized_residual` statistics: component count, mean,
second moment, and 90%/95% Gaussian marginal coverage (including coverage counts
and fixed thresholds). Controlled results group them by time; random-time
results additionally use the authored time bins. Velocity and endpoint have the
same standardized residual, so these statistics are calculated once in O(Bd)
streaming work. Coverage is per-coordinate, not a joint image confidence region;
conditional NLL is not generated-image likelihood. Good global summaries do
not prove calibration at every high-dimensional condition.

Reference-pair posterior diagnostics run once per checkpoint independently of
which sampler subsequently consumes it. Sampling-path maps have no paired
posterior residual labels and do not replace held-out calibration.

Posterior YAML explicitly owns `precision`, `compile`, `allow_tf32`,
`cudnn_benchmark`, and `num_workers` in
addition to batch size and statistical budgets. Use `--set precision=fp32
--set compile=false --set num_workers=4`. Precision/compilation share the same
execution validator and strict checkpoint loader as sampling; worker count
controls only data loading. These fields are required in authored YAML, so
an omitted execution choice cannot silently inherit the training mode.

`posterior_t_values` preserves authored order and values. Both configuration
preflight and direct diagnostics reject nonnumeric/nonfinite values, endpoints,
duplicates, FP32-rounded endpoints and distinct authored values that collapse to
the same FP32 time. Results cover every requested time exactly once. The `by_t`
key is Python's round-trip representation of the authored float (for example
`0.5` and `0.5000001`), also used for the time's independent noise seed. Display
rounding never determines result identity. Reordering or selecting a subset of
times preserves each retained time's noise and numerical result.

Compare consumer revisions as well as config digests: an unchanged config
cannot identify a changed implementation of its random protocol.

The canonical numerical result drives six chart families:

1. controlled variance and observed error by time;
2. controlled complete NLL by time;
3. random variance calibration;
4. random error-to-variance ratio;
5. random variance histogram;
6. random time calibration.

Component totals are streamed. Calibration and ranking retain bounded
per-image scalars only; spatial output is restricted to bounded sample
visualization. Equal predicted-variance ties are never split. Undefined
statistics use strict-JSON values with an explicit reason rather than NaN.
Raw variance, squared-error, and time observations are validated before any
accumulation or binning. Non-finite values, non-positive variance, negative
squared error, incompatible shapes, and out-of-range time fail the consumer;
they are never masked, filtered, or repaired. Suite, summary, calibration-bin,
and histogram counts must reconcile to the authored posterior population before
figures or artifacts are published.

The same diagnostic forward passes stream elementwise raw-logvar saturation
counts. Lower and upper saturation mean `sigmoid(raw_logvar) < 0.01` and
`sigmoid(raw_logvar) > 0.99`, respectively. Random-time results also report
these counts by authored time bin. Only per-image counts are retained for time
binning, no component tensor is persisted, and training performs none of this
diagnostic work.

Each suite, controlled time, and method owns independent stochastic state. A
change to controlled-time order or grid cannot alter random-time results, and
profiles never exchange noise, predictions, or accumulators. The baseline posterior seed is `321`; overrides are recorded as
a distinct evaluation protocol, as are changes to bin count or time grid.

## Sample visualization

`sample_visualization` is the only image-output profile. Acceptance uses seed
`321` and EMA weights. MNIST and CIFAR-10 publish eight samples per class;
ImageNet-256 publishes one per class. Any paper request remains bounded to at
most 1,000 images.

Sampling YAML owns inference `batch_size` independently of checkpoint training
batch size. It is required, positive, and overridable through `--set`; the
consumer binds the final configuration snapshot and its digest. Changing batch
size does not select new model weights or a different training protocol.

PFM velocity and endpoint variance maps are captured from the exact pre-update
state at the configured capture time (baseline `t=0.5`). Velocity scales come from the shared scientific log-variance
bounds; endpoint scales apply the corresponding factor
`(1-t)^2` (baseline `0.25`). Requests whose sampling grid cannot capture that time
exactly are invalid.

At the same `(x,t,condition)`, ODE and SDE read the same posterior maps. Their
visited states can differ, so trajectory maps can differ. SDE additionally
publishes `diffusion_rate = lambda(t) * Gamma_v` and
`brownian_step_covariance = h * lambda(t) * Gamma_v` with explicitly derived
scales. None of these maps denotes the final generated sample's conditional
variance. No full trajectories or terminal covariance matrices are retained.

## Shared sampling protocol

Both generated quality and visualization author the same closed `sampling`
mapping. Its required fields are `solver`, `steps`, `terminal_time`,
`terminal_readout`, `diffusion_scale`, `diffusion_schedule`,
`derivative_probes`, `precision`, `compile`, `allow_tf32`, and
`cudnn_benchmark`. Overrides support nested paths,
for example `--set sampling.steps=100 --set sampling.derivative_probes=8`.
Missing/unknown fields and overlapping overrides fail before execution.

ODE and SDE use continuous generation time, the uniform grid
`t_k = k * terminal_time / steps`, and the same terminal readout definition.
`state` returns the integrated state at the cutoff; `endpoint_mean` performs one
additional model evaluation there and requires a cutoff below one. Positive
SDE diffusion also requires `terminal_time < 1`; no hidden score denominator
clamp changes this contract. SDE requires a PFM checkpoint. ODE requires zero
diffusion scale. SDE with zero scale skips derivatives and Brownian draws and
uses the same ODE update under the same grid/readout.

`diffusion_schedule` is `constant` or `linear_decay`, where the latter means
`lambda(t) = diffusion_scale * (1-t)`. SDE uses the complete corrected Ito drift
and Euler--Maruyama update; a fixed number of independent Rademacher VJPs
estimates the effective variance Jacobian diagonal. The sigmoid bound and exp
remain in the differentiated graph. These inference choices do not alter the
checkpoint's source scale, variance bounds, interpolation, or training loss.
`precision` explicitly selects FP32 or BF16 inference; `compile` selects the
supported JiT or EDM compilation path. The model loader copies only these execution
settings, retaining the original training identity and weights. JiT EMA weights
are first loaded strictly into the recorded compile-wrapper layout; the known
block/head wrappers are then switched to the requested execution. This permits
compiled-checkpoint to eager inference without rewriting checkpoint keys or
relaxing strict loading.

The existing generic generation profiles retain their ODE/BF16 execution;
ImageNet profiles retain JiT compilation. The matched CIFAR profiles
`cifar10_edm_ode.yaml` / `cifar10_edm_sde.yaml` and corresponding sample profiles
use the same 50-step grid, cutoff 0.99, endpoint-mean readout, and BF16 compiled
execution. Their visualization capture is 0.495. The SDE scale 0.1, linear decay,
and two probes are explicit recommended settings, not a scientifically accepted
accuracy/cost optimum. Checkpoint probe and step refinement evidence is required
before treating any selected budget as validated.

Initialization retains global-index-seeded fixed blocks. Brownian increments
and derivative probes use separately seeded streams indexed by sample, step,
and probe where applicable. Changing batch partition or probe count cannot
change another stream's random inputs. Cross-device/kernel bitwise output
agreement is not promised. Brownian randomness belongs to the target SDE;
probe randomness belongs to its numerical estimate. For pathwise grid-refinement
comparisons, couple Brownian increments by summing fine increments; identical
seeds alone do not implement that coupling.

Results bind the full sampling protocol and distinguish per-chunk model forward
count (`actual_nfe`) from input VJP count (`actual_vjp`). Generated evaluation
also records total calls over chunks, sampler elapsed seconds, and peak device
allocated bytes during sampling. Elapsed time excludes checkpoint loading,
metric computation, and the initialization service; peak memory includes models
already resident on the device. These scopes are explicit in the result.

Bounded numerical summaries aggregate mean-square score, estimated divergence,
estimated drift correction, expected Brownian increment (`mean(h*lambda*gamma)`),
and the estimated derivative-induced update MSE. Chunk means are weighted by
sample count. The derivative error estimate is unavailable with one active
probe and is represented by null; estimated divergence/correction second
moments include Monte Carlo variation. Invalid coefficients or states fail the
sampler. A successful result records zero nonfinite values. Analytic solver tests
and controlled probe/step comparisons validate numerical behavior separately
from posterior calibration and generated quality.

Self-conditioning execution and its carry/refine policies are retired. The
fixed checkpoint identity `self_conditioning_mode: none` remains readable;
historical SC experiments use their original code revision. The same policy
applies to retired JiT SDPA execution. No new PFM-SDE training method identity is
introduced.

## Aggregation boundary

Authored comparison YAML names each consumer protocol with `owner` and `path`;
it does not maintain a manual digest. Preparation validates and normalizes the
referenced evaluation configurations, freezes them under `runs/inputs/configs`,
and binds their paths and SHA-256 values into the frozen comparison. An already
frozen reference includes its digest and must pass byte-integrity validation.
To compare a CLI-modified evaluation, reference its frozen input configuration
instead of the unchanged baseline YAML. Preparation never reloads a model.

Consumer manifest schema 2 uses `binding.config.sha256` as the sole consumer
configuration identity; the duplicate `consumer_protocol_sha256` binding field
is removed. Aggregate manifest schema 3 accompanies the per-profile
`consumer_config_sha256` summary. Checkpoint, dataset, training-comparison and
code provenance identities retain their separate purposes. Historical immutable
consumer artifacts are not rewritten or guessed into this schema; their
original revisions remain available. The public checkpoint runtime identity
contains project name, source revision, rank and world size. Historical
deployment checkpoints require their producer revision; this release does not
silently migrate their metadata.

Aggregate consumes explicit compact numerical result references plus a closed
comparison definition. Only `generated_common` and `posterior_diagnostics` may
be numerical comparison dependencies; `sample_visualization` is independent
qualitative evidence and never blocks numerical completion. Aggregate validates
complete method/profile coverage and shared protocols, then requires every input
to bind the explicitly requested checkpoint epoch and deterministically
summarizes canonical numerical owners. The shared checkpoint step remains
derived evidence rather than an operator input. Aggregate does not load
checkpoints, datasets, generated images, or discover results by scanning run
trees. One dependency-complete aggregate is the only fan-in for a numerical
comparison. The current method/profile coverage key admits one generated
protocol per comparison; ODE and SDE use separate bound artifacts/comparisons
without inventing different trained-method identities.

`configs/comparisons/cifar10_sde.yaml` defines PFM Full/Delta SDE quality plus
their solver-independent posterior diagnostics. The four-method
`cifar10_primary.yaml` remains the ODE baseline. For a matched cutoff/readout ODE
comparison, override its generated protocol reference with
`configs/evaluation/cifar10_edm_ode.yaml`; the SDE comparison uses the matched
`cifar10_edm_sde.yaml`. Baseline FM methods never become SDE-capable by changing
a comparison label. Each comparison still requires its complete explicit input
set; bounded sample visualizations remain independent. See
[reproduction](reproduction.md) for checkpoint selection and protocol scope.

The aggregate manifest records the exact Git revision that executed publication,
separately from the input `consumer_revision`; the revisions may differ when a
newer aggregator deliberately rebuilds older immutable consumer results.
`aggregate_revision`, `consumer_revision`, and `checkpoint_revision` are
producer provenance, not format versions. Within one comparison the consumer
revisions must agree with one another and the checkpoint revisions must agree
with one another, but those two groups and the aggregate executor need not be
equal. Publication validates the current aggregate-manifest shape and
comparison identity in memory, then binds the digest of the exact result bytes
written before immutable publication.

## Bounded checkpoint SDE validation

Run the CUDA diagnostic from Python with an explicit checkpoint milestone:

```bash
python -m pfm.cli.sde_acceptance \
  --variant-root ./outputs/experiments/cifar10/pfm_delta --epoch 600 \
  --diagnostic-id sde-check --config configs/diagnostics/cifar10_sde_acceptance.yaml \
  --data-root ./data --output-dir ./outputs --device cuda
```

It uses the normal frozen YAML and `--set` protocol. The ImageNet workload is
`configs/diagnostics/imagenet256_sde_acceptance.yaml`. Both use the checkpoint's
PFM variant, architecture, source scale, shape, original variance bounds and
canonical held-out dataset; no legacy checkpoint branch or parallel sampler is
introduced. A seeded held-out subset is recorded by index and label.

The bounded fields are batch size, seed, probe repetitions, reference times,
probe counts, precisions, compile, TF32, cuDNN benchmark, sampler steps/probes, diffusion scale,
reference terminal time and cutoff times. Missing/unknown fields, nonfinite
values, duplicate budgets and incompatible refinement grids fail in preflight.
The largest step budget must be divisible by every coarser one. Step/probe
comparisons at a shared cutoff use coupled Brownian increments. Cutoff
comparisons share normalized random inputs but do not claim identical physical
Brownian paths. Reports bind the authored config digest and checkpoint lineage.

Matched ImageNet visualization profiles are
`configs/sampling/imagenet256_ode_visualization.yaml` and
`configs/sampling/imagenet256_sde_visualization.yaml`: 1000 classes, one image
per class, 250 steps, cutoff .99, endpoint-mean readout, BF16/compile, capture .495.
The SDE candidate uses two probes and .1 linear-decay diffusion. Batch size is
independent of output population. These protocols require exact-revision CUDA
validation; a configuration file alone does not establish accuracy or cost.

EDM uses in-place module compilation: parameter identities and state-dict keys
are unchanged. Consumer execution is selected independently before strict EMA
loading. The old EDM checkpoint configuration lacking `edm_compile` records
necessarily eager execution; only the checkpoint reader recognizes that one
prior representation. New authored training YAML must contain the field.
ImageNet quality uses `imagenet256_jit_ode.yaml` and `imagenet256_jit_sde.yaml`,
which match the visualization protocol and metric backend.
