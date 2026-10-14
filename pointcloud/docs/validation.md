# CPU verification

Verified on 2026-10-13 with macOS arm64, Python 3.12.14 and PyTorch 2.8.0,
using two CPU threads.

## Automated checks

Six tests cover Gaussian NLL and gradients, exact pointwise divergence, joint
encoder/prior gradients and flow inversion, ODE/SDE endpoint readouts, native
Chamfer metrics, and training/resume/sample/evaluate/diagnostics. The workflow
copies a checkpoint to another path before resuming; model tensors, Adam states,
scheduler and Torch RNG match uninterrupted training bit for bit.

```bash
python -m pip install '.[test]' build
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python -m build
```

## Installed package and real data

The built wheel was installed and run outside the checkout. All four commands
work without repository imports. Using the official Chair initialization, actual
ShapeNet HDF5 data and full native network (3,885,360 parameters, 14 prior layers,
2,048 points):

- Four updates on four Chair shapes completed. A separate two-update run resumed
  from a copied checkpoint in another directory; model, optimizer, scheduler and
  Torch RNG matched the uninterrupted run bit for bit.
- Both ODE and SDE generated samples and PNG galleries. Native CD metrics and
  independent posterior diagnostics completed on two held-out Chair shapes.
- A retained trained Airplane checkpoint completed both samplers and native CD
  evaluation on two real held-out shapes.

## Algorithm preservation

With identical weights, input and RNG, the simplified implementation matched the
previous public implementation exactly for a native-size joint training update,
all updated model tensors, ODE/SDE samples, native CD metrics, and real dataset
ordering and normalization. The retained pointwise sampler, PointNet encoder,
flow prior and metric definitions are unchanged.

These checks verify the CPU code path and preservation of the main algorithm;
they do not reproduce paper quality. No GPU, distributed training, full training
or paper-scale metric run was performed. The Airplane initialization provider
previously returned a download-quota limit, so fresh initialization was exercised
with Chair; Airplane used an existing research checkpoint. Weights are not bundled.
