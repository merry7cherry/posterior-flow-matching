# PFM PushT

Maintain one PFM workflow: native observation encoding, joint Gaussian NLL,
ODE/SDE action plans, closed-loop PushT evaluation and reference diagnostics.
Preserve the trainable crop84/ResNet/GroupNorm encoder, action horizon16 with
executed slice1:9, native normalization, bounded variance, derivative correction,
EMA and ordinary optimizer/scheduler recovery. Keep the implementation focused
on this workflow.

Run CPU tests with at most two threads. Keep datasets, checkpoints and test
outputs outside the source tree. Keep README commands aligned with the entrypoints.
Preserve upstream license texts and attribution.
