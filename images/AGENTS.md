# PFM image task

This directory is the image task inside the PFM monorepo. Work from `images/`
with its own Python environment. `README.md` describes the public commands.

Keep one PFM implementation: a configurable velocity or endpoint mean head,
velocity variance, Gaussian NLL, corrected ODE/SDE, FID/IS and held-out raw
moment/NLL diagnostics. Preserve native DDPM++ and JiT architecture and licenses.
MNIST is the small CPU example. Keep the training loop, ordinary checkpoint
save/resume and EMA direct and readable. Do not reintroduce experiment binding,
source identities, immutable artifacts, aggregation, variant registries, locks,
standardized-residual coverage, cluster submission scripts or performance campaigns.

Test mathematical changes and the affected execution path proportionately on CPU.
Keep temporary run outputs outside the repository.
