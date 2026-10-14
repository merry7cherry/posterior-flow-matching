# CPU verification

The simplified monorepo package was checked on macOS arm64, Python 3.10.20,
PyTorch 2.7.1, with at most two CPU threads. These are local CPU observations;
the repository CI reports Linux results separately.

## Mainline preservation

Against the previous public PushT implementation at commit
`56ee7790d948a6c5ba439bc11a03c459582a1484`, fixed weights and inputs give exactly
identical joint NLL, every parameter gradient, one AdamW update, learning rate,
EMA weights, ODE/SDE action plans, endpoint variance, field outputs and exact
input variance derivatives. This comparison uses the native image encoder and
a reduced-width `[16,32]` action U-Net; the variance head is nonconstant.

The full `[512,1024,2048]` native network still has 262,710,070 parameters. A
CPU batch1 forward/backward with synthetic images/actions produced finite NLL
and gradients in the encoder and both heads. This is a differentiation check.

The original and simplified data paths agree exactly on the official 206-episode,
25,650-frame dataset: train/validation indices, sampled windows and normalization.
The native simulator's fixed-seed reset, one-step observations and reward also
agree exactly. The retained state assignment order matches the demonstrations;
there is no alternative simulator state-order switch in this package.

## Workflow checks

- 13 CPU tests passed: NLL, bounded variance derivatives, exact/stochastic VJP,
  Gaussian SDE moments, ODE/zero-diffusion agreement, native observation/action
  interfaces, training gradients, model/EMA interruption recovery, parallel
  simulator parity and video production.
- Two single-batch epochs completed on the official local demonstrations using
  the reduced smoke U-Net, with native rollouts, videos and saved checkpoints.
- Real-data interrupted-versus-uninterrupted model/EMA recovery passed separately.
  That test substitutes a no-op runner; native physics is checked independently.
- Wheel and source archives built with full third-party license texts. An isolated
  environment installed the wheel; all three CLIs ran outside the checkout.
  ODE/SDE rollout videos and standalone reference diagnostics were produced from
  the saved real-data smoke model. Diagnostic evaluation does not launch a rollout.

The synthetic storage fixture used by ordinary tests is explicitly separate from
the official-data checks. The new payload stores explicit config, model/EMA,
optimizer and resume state; no original output directory or source identity is
required to load it.

After simplifying logging to append-only scalar JSONL, all 13 CPU tests passed
again. An additional official-data check used two updates per epoch, interrupted
after the first completed-epoch checkpoint, and resumed in the same output
directory. Existing log records remained intact and subsequent records appended;
the final model and EMA weights exactly matched uninterrupted training. Logging
no longer repairs partial records, reads prior values, or reports elapsed time
and peak CUDA memory.

The cleanup wheel and source distribution were rebuilt on 2026-10-14 UTC. All
48 tracked runtime/configuration files matched the wheel, and all three installed
CLI help commands ran outside the checkout.

## Re-run

```bash
OMP_NUM_THREADS=2 pytest -q
PFM_TEST_DATA=/absolute/path/pusht_cchi_v7_replay.zarr pytest -q tests/test_resume.py
uv build --wheel
```

See the README for training, rollout and independent diagnostic commands.
No GPU training, full optimization run, timing benchmark, trained-weight release
or reproduction of paper quality metrics is claimed. The pinned upstream Gym and
torchvision versions emit deprecation notices during these checks.
