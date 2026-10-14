# Data and category initialization

Use the assets linked by the [official upstream repository](https://github.com/luost26/diffusion-point-cloud#datasets-and-pretrained-models).
Download them to a directory of your choice; this repository never downloads data
or checkpoints automatically and does not redistribute them. Review the original
ShapeNet dataset terms and the asset provider's terms before obtaining them.

| File | Provider link |
|---|---|
| ShapeNet HDF5 | [shapenet.hdf5](https://drive.google.com/file/d/1WdooIyOMVt_HVd_A0QOM9DTTX41O65jQ/view) |
| Chair encoder/prior initialization | [GEN_chair.pt](https://drive.google.com/file/d/1qXtJCOevgdyhxwllJ5G4e7ZGH0sFsDXF/view) |
| Airplane encoder/prior initialization | [GEN_airplane.pt](https://drive.google.com/file/d/1NTVl1RKcIJDildrTx-jUpBJkDNzg55ld/view) |

The HDF5 groups are `03001627` (Chair) and `02691156` (Airplane), each with
`train`, `val`, and `test` arrays of `[shapes,2048,3]`. Chair has 5,602 train and
989 test shapes; Airplane has 3,438 train and 607 test shapes. The public pipeline
uses **train and test only**, preserving upstream splits. The supplied validation
split contains exact-array overlap with train/test; it is not used for checkpoint
selection. The training probe is explicitly drawn from training data.

Training subtracts each shape's mean and divides by its flattened sample standard
deviation (`shape_unit`). Evaluation separately centers and scales every generated
and reference cloud by the maximum bounding-box half-extent (`shape_bbox`).
Category data are ordered by original index then shuffled with Python seed 2020,
matching the native loader. A subset uses the prefix of that order.

Initialization imports only the category's PointNet encoder and flow-prior
weights. If an upstream prior stores spectral-normalized weights, initialization
imports the effective linear weights once. Training then uses ordinary parameters.
The coordinate PFM is initialized separately. Sampling checkpoints carry
all three models and do not depend on `GEN_*.pt`. Resume reads the saved configuration and model directly from the checkpoint.
