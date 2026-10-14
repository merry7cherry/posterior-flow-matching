# PFM public implementation

This repository contains one PFM project with four parallel implementations:
`images/`, `pointcloud/`, `molecules/`, and `pusht/`. `website/` hosts their shared
project page. Each task owns its dependencies, configuration, native models,
training, sampling, evaluation, diagnostics and CPU tests.

Keep only the paper's algorithm and task necessities. Prefer direct checkpoint
and result files. Do not add experiment registries, identity/hash binding,
immutable-artifact protocols, directory ownership locks, historical compatibility
layers, non-PFM training branches or campaign/acceptance frameworks.

Preserve the native mathematical behavior: Gaussian NLL, variance parameterization,
corrected ODE/SDE dynamics, endpoint readouts and task-specific geometry/conditioning.
Use separate task environments and CPU checks by default. CPU execution does not
establish GPU or paper-scale performance. Keep weights and datasets out of Git.

Update the affected task README and existing verification record when changing an
entrypoint. The root README is the four-task code index; update page links with it.
Keep upstream licenses and attribution with the code they cover.
