# CPU verification

Automated checks reverified on 2026-10-14 UTC with macOS arm64, Python 3.12.14 and PyTorch 2.8.0,
using two CPU threads.

## Automated checks

Seven tests cover Gaussian NLL and gradients, exact pointwise divergence, joint
encoder/prior gradients and flow inversion, upstream spectral-weight import,
ODE/SDE endpoint readouts, native Chamfer metrics, and
training/resume/sample/evaluate/diagnostics. The workflow
copies a checkpoint to another path before resuming; model tensors, Adam states,
scheduler and Torch RNG match uninterrupted training bit for bit.

```bash
python -m pip install '.[test]' build
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python -m build
```

## Installed package and real data

The cleanup wheel and source distribution were rebuilt; all 15 tracked runtime
files matched the wheel, and all four installed CLI help commands ran outside
the checkout.

The 2026-10-13 built wheel was installed and run outside the checkout. All four commands
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

The 2026-10-14 cleanup replaced upstream spectral-normalization hooks with direct
initialization-weight materialization and removed unused partial-chain execution.
Against the previous loader, the actual Chair asset produced identical model
tensors, joint loss and gradients, latent density/inversion, and ODE/SDE outputs.
A constructed spectral-normalized asset also matched these paths, with only
floating-point rounding differences (maximum absolute difference below 2e-7).
The import test additionally checks effective weights against an SVD reference
and confirms that no runtime normalization hooks remain.

These checks verify the CPU code path and preservation of the main algorithm;
they do not reproduce paper quality. No GPU, distributed training, full training
or paper-scale metric run was performed. The Airplane initialization provider
previously returned a download-quota limit, so fresh initialization was exercised
with Chair; Airplane used an existing research checkpoint. Weights are not bundled.
