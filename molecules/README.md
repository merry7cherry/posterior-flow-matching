# Posterior Flow Matching — Molecules

Standard PFM for GEOM-QM9 conformer generation: learn a velocity mean and atomwise variance with the native ET-Flow SO(3) network, then sample with ODE/SDE and inspect endpoint posterior readouts.

[All tasks](../README.md) · [Images](../images/) · [Point clouds](../pointcloud/) · [PushT](../pusht/) · [Project page](https://merry7cherry.github.io/posterior-flow-matching/) · [Paper PDF](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf)

## Install

Run from this `molecules/` directory in a separate Python 3.11 environment. Linux CPU:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
python -m pip install '.[test,visualize]'
```

Apple Silicon CPU uses the normal PyTorch wheel and the matching PyG extension. With Xcode Command Line Tools installed:

```bash
python -m pip install torch==2.5.1 numpy==1.26.4 setuptools wheel
CXXFLAGS=-Wno-invalid-specialization MAX_JOBS=2 python -m pip install torch-cluster==1.6.3 --no-build-isolation
python -m pip install '.[test,visualize]'
```

The Clang flag handles a PyTorch 2.5 header diagnostic on recent macOS SDKs. No trained weights or datasets are included. CPU evidence and its limits are in [VALIDATION.md](VALIDATION.md).

## Quick CPU example

The example generates small RDKit fixtures, not GEOM benchmark data. The smoke configuration uses the native network with reduced width/depth.

```bash
pfm-molecules-example --output-dir data/example
pfm-molecules-train --config configs/cpu-smoke.yaml --data-dir data/example --run-dir runs/example --cpu
pfm-molecules-sample --checkpoint runs/example/checkpoints/final.ckpt --data-dir data/example --output runs/example/generated.pt --mode ode --device cpu
pfm-molecules-evaluate --input runs/example/generated.pt --output-dir runs/example/metrics
pfm-molecules-visualize --input runs/example/generated.pt --output runs/example/conformer.png
```

These commands check the workflow; their scores are not benchmark results. Installed commands work outside the checkout when given explicit paths.

## Prepare GEOM-QM9

Obtain raw QM9 conformer pickles from [GEOM](https://dataverse.harvard.edu/api/access/datafile/4327252), and `split.npy` / `test_mols.pkl` from the [Torsional Diffusion instructions](https://github.com/gcorso/torsional-diffusion#training-model). Keep the complete raw set: split indices refer to sorted `*.pickle` filenames. The native input convention uses a consistent atom ordering across each molecule's conformers. Dataset terms are separate; only load trusted pickle/checkpoint files.

```bash
pfm-molecules-prepare --raw-dir /path/to/rdkit_folder/qm9 --metadata-dir /path/to/QM9 --output-dir data/processed
```

Use a fresh output directory. Preparation preserves supplied validation/test molecules and excludes matching molecular graphs from training. Training/validation retain at most 30 conformers ranked by the supplied Boltzmann weights. Output is `qm9/{train,val,test}/*.pt` plus `counts.json`.

## Train and resume

```bash
pfm-molecules-train --config configs/qm9.yaml --data-dir data/processed --run-dir runs/qm9 --cpu
pfm-molecules-train --config configs/qm9.yaml --data-dir data/processed --run-dir runs/qm9 --resume runs/qm9/checkpoints/last.ckpt --cpu
```

`qm9.yaml` retains the native 20-layer, 160-channel network. Set batch size, epochs and optimization before starting. Checkpoints contain model, optimizer, scheduler, epoch/global step, configuration and RNG states. Ordinary resume continues from the last saved epoch; keep the same configuration and planned schedule. `--stop-after-epoch N` can exercise an interruption while keeping that schedule.

The loss is one joint Gaussian NLL on the centered `3(N-1)`-dimensional subspace. [ARCHITECTURE.md](ARCHITECTURE.md) connects the implementation to the molecular geometry.

## Sample and evaluate

```bash
pfm-molecules-sample --checkpoint runs/qm9/checkpoints/final.ckpt --data-dir data/processed --output runs/qm9/ode.pt --mode ode --device cpu
pfm-molecules-sample --checkpoint runs/qm9/checkpoints/final.ckpt --data-dir data/processed --output runs/qm9/sde.pt --mode sde --device cpu
pfm-molecules-chirality --source runs/qm9/sde.pt --output runs/qm9/sde-corrected.pt
pfm-molecules-evaluate --input runs/qm9/sde-corrected.pt --output-dir runs/qm9/metrics --workers 1
```

Sampling generates twice the reference conformer count, using 50 steps and endpoint-mean completion at `t=0.99` by default. SDE uses `lambda(t)=0.3*(1-t)` in both drift and noise. `--steps`, `--diffusion-scale` and `--max-molecules` select smaller or changed runs. Each call writes its requested output directly.

`generated.pt` is a list of native PyG `Data` objects: `smiles`, `rdmol`, `pos_ref`, `pos_gen`, and per-atom `endpoint_variance` for the first generated conformer. Read it with `torch.load(path, weights_only=False)`.

Evaluation retains heavy-atom, symmetry-aware RMSD and writes `metrics.json` / `coverage.csv`: COV-R/COV-P are higher-is-better, MAT-R/MAT-P lower-is-better (angstroms). Coverage is summarized at 0.5 Å. The native evaluator excludes disconnected inputs and performs no force-field relaxation.

Generation retains the existing ET-Flow atom encoding: its chirality feature maps all tetrahedral tags to the same value, so this feature does not distinguish input stereoisomers for the network. Input stereochemistry is used by the optional `pfm-molecules-chirality` postprocessing command. It reflects a whole conformer only when every specified tetrahedral center disagrees with the input stereo; it leaves partial disagreements unchanged and never uses reference RMSD to select a correction. This is the existing generation-and-postprocessing pipeline, not a learned chirality-conditioning claim. Save corrected outputs separately when retaining raw samples.

## Posterior diagnostics

```bash
pfm-molecules-diagnostics --checkpoint runs/qm9/checkpoints/final.ckpt --config configs/diagnostics.json --data-root data/processed --output runs/qm9/diagnostics --device cpu
pfm-molecules-visualize --input runs/qm9/sde.pt --output runs/qm9/conformer.png
```

Diagnostics fit variance bins and a time-only covariance baseline on validation reference interpolations, then measure grouped residual second moments and Gaussian NLL on test reference interpolations. Molecules receive equal weight. The atom-averaged first residual moment is identically zero under centering and is not reported as a diagnostic. `design.json` contains the validation design; `summary.json` contains test results and molecule-bootstrap intervals. These checks apply to the raw model before chirality postprocessing and do not establish full conditional calibration or final-sample risk. The visual example colors the first conformer by endpoint covariance trace. After stereo correction, those variance values remain raw-model readouts.

## Tests and attribution

```bash
python -m pytest -q
python -m build
```

See [VALIDATION.md](VALIDATION.md) for executed checks. CUDA, distributed training, speed and paper-scale reproduction have not been validated. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records ET-Flow and TorchMD-Net attribution. Cite **Posterior Flow Matching** ([CITATION.cff](CITATION.cff)) and [ET-Flow](https://github.com/shenoynikhil/ETFlow). Code is MIT with upstream notices in [LICENSE](LICENSE) and [LICENSES](LICENSES).
