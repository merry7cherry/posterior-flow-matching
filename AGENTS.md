# PFM public implementation

This repository contains one PFM project with four parallel implementations:
`images/`, `pointcloud/`, `molecules/`, and `pusht/`. `website/` hosts their shared
project page. Each task owns its dependencies, configuration, native models,
training, sampling, evaluation, diagnostics and CPU tests.

Keep only the paper's algorithm and task necessities, with direct checkpoint
and result files. Keep public documentation focused on the method and runnable
workflows; do not add development audit or verification reports.

Preserve the native mathematical behavior: Gaussian NLL, variance parameterization,
corrected ODE/SDE dynamics, endpoint readouts and task-specific geometry/conditioning.
Use separate task environments and proportionate CPU tests. Keep weights,
datasets and temporary outputs out of Git.

Update the affected task README when changing an entrypoint. The root README is
the four-task code index; keep project-page links and descriptions aligned with it.
Keep upstream licenses and attribution with the code they cover.
