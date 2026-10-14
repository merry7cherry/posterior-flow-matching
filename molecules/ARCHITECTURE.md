# Molecular PFM mainline

The user workflow is prepare → train/resume → sample → evaluate → diagnose. CLI modules read and write ordinary files. `sample.py` writes a list of PyG `Data` records to `generated.pt`; `evaluate.py`, `chirality.py`, and `visualize.py` consume that list directly.

`etflow/models/pfm_model.py` uses the native ET-Flow SO(3) backbone in `models/model.py` and `networks/torchmd_net/`. Molecular input features, bond/radius graph extension and equivariant representations are retained. A shared scalar head predicts one bounded log-variance per atom, repeated over its Cartesian coordinates. The mean velocity is centered per molecule.

For `P = I - 11ᵀ/N`, velocity covariance is `Gamma_v = P diag(g) P` on the centered coordinate subspace (with an identity over Cartesian coordinates). `pfm_probability.py` implements covariance products, projected noise, marginal variances and the subspace Gaussian NLL. This single joint training objective is normalized by `3(N-1)` and omits parameter-independent constants.

`pfm_sampling.py` integrates either `v` or the corrected Ito drift `v + lambda(t)/2 * (div_H Gamma_v + Gamma_v score)`, with noise `sqrt(lambda(t)) P diag(sqrt(g)) dW`. The independent-Gaussian-source score is `(t*v-x)/(source_scale²*(1-t))`. Covariance divergence uses atomwise Rademacher VJPs, projecting before and after the sign product. Tests compare it against the exact covariance Jacobian. No integration-step graph is retained.

The SDE uses the manuscript's linear schedule `lambda(t)=lambda_0*(1-t)` (default `lambda_0=0.3`). Endpoint readouts are `mean_x=x+(1-t)*v` and `Sigma_x=(1-t)²*Gamma_v`. Both samplers complete with the endpoint mean at finite terminal time. ODE and zero-noise SDE coincide.

`cli/train.py` uses Lightning, AdamW and a single warmup/cosine schedule over the configured run. The small `TrainingState` callback saves configuration/RNG states and controls deterministic epoch-level conformer selection. Lightning saves model, optimizer, scheduler and progress; `last.ckpt` resumes an ordinary run. There is no multi-stage training path.

`commons/covmat.py` retains native symmetry-aware heavy-atom RMSD and COV/MAT. `cli/chirality.py` optionally applies input-stereo whole-conformer reflection. `cli/diagnostics.py` learns bins and the time-only baseline on validation reference states, then evaluates held-out test residual moments and NLL. It preserves molecular geometry and equal-molecule weighting without introducing a separate result-management protocol.
