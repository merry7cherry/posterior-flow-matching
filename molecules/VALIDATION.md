# CPU verification

Validated 2026-10-13 on macOS Apple Silicon with CPython 3.11.15, PyTorch 2.5.1, PyG 2.6.1, torch-cluster 1.6.3, Lightning 2.5.0.post0, RDKit 2024.9.6 and NumPy 1.26.4. Computation used CPU only with at most two PyTorch threads.

- **22 tests passed.** They cover projected Gaussian NLL/gradients against an explicit subspace basis; covariance/noise; exact covariance-Jacobian divergence and Rademacher expectation; JVP/VJP duality; ODE/zero-noise SDE equivalence; Gaussian-path sampling; the linear drift/noise schedule; native SO(3) forward/backward and proper-rotation equivariance; native metrics; input-stereo correction; held-out split preparation; posterior diagnostics; and ordinary training recovery.
- Full original-public-vs-simplified comparison uses the **20-layer, 160-channel, 9,090,130-parameter** native model, identical weights and two actual GEOM molecules. Velocity, log-variance, input gradients, projected divergence, joint NLL, all parameter gradients, ODE/SDE outputs, endpoint variances and optimizer LR schedule agree exactly (maximum absolute difference 0). SDE comparison explicitly uses `lambda(t)=0.3*(1-t)` on both sides.
- The training recovery test compares uninterrupted and interrupted/resumed two-epoch runs. Weights, optimizer tensors, scheduler state and global step agree exactly.

- A fresh isolated environment installed the built wheel and its declared dependencies from the local package cache. From an arbitrary directory outside the checkout, installed CLI commands completed actual GEOM smoke training, interruption/resume, continuous training, ODE/SDE sampling, stereo correction, native COV/MAT, held-out posterior diagnostics and a rendered PNG example. The same **22 tests passed against the installed package**.
- The actual-data smoke split contains four train, two validation and two test molecules, one conformer each. It is a custom small split, not the official benchmark. The reduced 72.5K-parameter native model trained for two epochs/four optimizer steps; saved weights, optimizer tensors, scheduler state and global step match its uninterrupted run exactly. Full-width scientific parity is the separate comparison above.
- Wheel/sdist packaging includes upstream license texts, configurations, tests and documentation. Installed commands run independently of the source checkout.

Synthetic RDKit fixture metrics are only workflow checks. These CPU checks do not establish CUDA, BF16, DDP, full training, benchmark performance or timing claims. No trained weights are distributed.
