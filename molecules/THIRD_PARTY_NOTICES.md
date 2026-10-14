# Third-party notices

The PFM molecular implementation derives from [ET-Flow](https://github.com/shenoynikhil/ETFlow), upstream base `80f82d7d142bc3872412d2966a8d71ed14462b18`. ET-Flow's MIT license and author copyright are preserved in [LICENSES/ETFlow-MIT.txt](LICENSES/ETFlow-MIT.txt) and the root LICENSE. The reviewed research-source snapshot was `881dc08bbee7cfa25d48a41e7456b3c3044464f1`; this release has independent clean Git history.

The files under `etflow/networks/torchmd_net/` retain the ET-Flow adaptation of the [TorchMD-Net](https://github.com/torchmd/torchmd-net) equivariant transformer. Its MIT notice is preserved in [LICENSES/TorchMD-Net-MIT.txt](LICENSES/TorchMD-Net-MIT.txt). Copyright (c) 2021–2023 Universitat Pompeu Fabra.

`etflow/commons/`, `etflow/data/`, the active molecular backbone constructor/forward, and the COV/MAT evaluator retain or adapt ET-Flow code and conventions. PFM adds the variance head, projected Gaussian objective, covariance-aware sampler, diagnostics, portable CLI and runtime checks. Non-PFM priors, upstream default checkpoints and unrelated tasks were removed; upstream authorship is not reassigned.

PyTorch, PyTorch Geometric/torch-cluster, Lightning, RDKit, Datamol and other dependencies are installed separately and retain their licenses. GEOM data and Torsional Diffusion split/reference files are fetched independently under their respective terms; neither datasets nor trained weights are included here. Links in README identify the authoritative data sources.
