# PFM PushT

Maintain one PFM workflow: native observation encoding, joint Gaussian NLL,
ODE/SDE action plans, closed-loop PushT evaluation and reference diagnostics.
Preserve the trainable crop84/ResNet/GroupNorm encoder, action horizon16 with
executed slice1:9, native normalization, bounded variance, derivative correction,
EMA and ordinary optimizer/scheduler recovery. Keep the code direct; remove unused
modes and historical compatibility machinery instead of adding frameworks.

Use CPU verification with at most two threads. Keep datasets/checkpoints and QA
artifacts outside the source tree. Preserve upstream license texts and attribution.
Update README/VERIFICATION when entrypoints change. CPU checks do not establish
GPU execution, full training convergence or paper-level performance.
