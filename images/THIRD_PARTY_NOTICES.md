# Third-party sources and licenses

The licenses below apply to their respective upstream components. They do not
grant a license for datasets, pretrained weights, or paper assets. Original
PFM code in this image task is licensed under the MIT License
in [LICENSE](LICENSE). The EDM-derived file, including its local adaptations,
is instead distributed under CC BY-NC-SA 4.0.

## JiT transformer

`src/pfm/models/jit_backbone.py` adapts the transformer architecture and utility
operations from [LTH14/JiT](https://github.com/LTH14/JiT), revision
`cbc743a2ada5e9762697da2c83f8c4f8379e8c17`. JiT is copyright (c) 2025 Tianhong Li
and licensed under the MIT License. The full upstream license is reproduced in
[`third_party/licenses/JiT-MIT.txt`](third_party/licenses/JiT-MIT.txt).

PFM uses local prediction heads, configuration validation, device-independent
tensor creation, and explicit inference execution controls. The training
objectives and posterior interpretation belong to PFM. JiT itself acknowledges
[SiT](https://github.com/willisma/SiT) and
[LightningDiT](https://github.com/hustvl/LightningDiT).

## EDM DDPM++ backbone

`src/pfm/models/edm_backbone.py` ports the DDPM++ backbone structure described by
[NVlabs/edm](https://github.com/NVlabs/edm). The upstream network source and
license were checked at revision `008a4e5316c8e3bfe61a62f874bddba254295afb`;
this identifies the reviewed upstream snapshot, not a claim that this was the
original port revision.

The upstream source is copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES and
licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International](https://creativecommons.org/licenses/by-nc-sa/4.0/). Its license
and warranty disclaimer are preserved in
[`third_party/licenses/EDM-CC-BY-NC-SA-4.0.txt`](third_party/licenses/EDM-CC-BY-NC-SA-4.0.txt).
The local port uses PyTorch modules, exposes a feature map and conditioning
vector, and connects to PFM prediction heads rather than EDM preconditioning.
Its upstream restrictions must be retained in redistribution.

## FID evaluation dependency

FID evaluation uses [LTH14/torch-fidelity](https://github.com/LTH14/torch-fidelity)
at `bfe72b995473e4fee9351705d7b005f17442a4ca`, installed through the `eval` extra.
This fork is part of the evaluation protocol; a different implementation is
not an equivalent drop-in metric backend. The dependency is copyright 2020
Anton Obukhov and distributed under Apache License 2.0. Its upstream notice is
copied in
[`third_party/licenses/torch-fidelity-NOTICE.txt`](third_party/licenses/torch-fidelity-NOTICE.txt).
The dependency source is not vendored into this repository.

## Datasets and model assets

MNIST, CIFAR-10, and ImageNet retain their respective dataset terms. ImageNet
must be obtained through its own access process. Inception weights and any
separately distributed model weights retain their source licenses; repository
availability does not supersede those terms.
