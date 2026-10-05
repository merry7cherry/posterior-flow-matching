# PFM implementation guidance

This repository owns the PFM image implementation, dependencies, tests,
configurations and public project page. Follow README.md, docs/architecture.md
and docs/reproduction.md for the current software and scientific contracts.

## Scope

Keep changes focused on the paper's methods, training, sampling, evaluation
and reproducibility. Preserve the existing probability/model/training/sampling
boundaries. Do not add scheduler-specific deployment, environment replacement,
historical migration frameworks or duplicate command implementations.
The molecular, point-cloud and PushT implementations are separate repositories;
inspect their instructions and Git state before changing them.

## Runtime and evidence

The public interface uses explicit data/output paths, Python and standard
torchrun. Training defaults to CUDA. CPU is an explicit small-example/test
path; never silently substitute hardware, precision, sampler or model settings.
A complete YAML and explicit --set overrides define an experiment. Batch size
is per rank. Changes to world size, batch size, precision or workers change the
training/resume contract and must be reported.

Record source revision when known, effective configuration, checkpoint/data
identity, seeds and actual evaluation protocol. Unknown or dirty source must
not be labeled an exact clean revision. Preserve historical producer identity.
Local checks prove only what they exercise. CUDA, compiled derivatives, DDP and
paper-scale claims require evidence from the corresponding execution. A short
run proves software behavior, not converged generation quality.

## Changes and publication

Make the smallest coherent change, preserve unrelated edits and run relevant
checks. Update the affected README or architecture/reproduction documentation
when an entrypoint or durable contract changes. Keep run evidence with its
outputs; do not create a parallel project status registry.

Use the uva-hpc skill if explicitly operating on UVA infrastructure, and honor
its project/account boundaries without reintroducing that infrastructure into
the public codebase. Pushes, visibility changes, deployment, job submission and
external data transfers require user authority when not already covered by the
request. Verify the actual external result; local preparation is not deployment.

Before a public release, verify licensing and upstream attribution, the exposed
content/history, real documentation commands and website links. Do not invent
weights, publication status, paper links or empirical results.
