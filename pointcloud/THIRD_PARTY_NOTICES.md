# Third-party notices

- **Diffusion Probabilistic Models for 3D Point Cloud Generation**,
  Shitong Luo and Wei Hu, CVPR 2021:
  <https://github.com/luost26/diffusion-point-cloud>. MIT license, copyright
  (c) 2021 Shitong Luo, preserved in `LICENSE`. The PointNet encoder, coupling
  flow, spectral-normalization asset compatibility, pointwise coordinate backbone,
  context layers, schedule, normalization and metric routines derive from this
  implementation. PFM replaces its diffusion training and sampling; the diffusion
  decoder is not distributed here.
- **PointFlow**, <https://github.com/stevenygd/PointFlow>:
  MIT license, copyright (c) 2019 Guandao Yang. Native metric routines in
  `pfm_pointcloud/metrics.py` descend from its metrics. Full notice is preserved
  in `licenses/PointFlow-LICENSE`.
- **AtlasNet**, <https://github.com/ThibaultGROUEIX/AtlasNet>:
  MIT license, copyright (c) 2019 ThibaultGROUEIX. The native Chamfer-distance
  routine is attributed by the upstream implementation to AtlasNet. Full notice
  is preserved in `licenses/AtlasNet-LICENSE`.
- The upstream k-nearest-neighbor metric credits
  <https://github.com/xuqiantong/GAN-Metrics>; that attribution is retained in the
  source. The routine is redistributed as part of the MIT-licensed PointFlow
  and diffusion-point-cloud derivative.

The source snapshot was extracted from PFM research revision
`d3d119386c06da9dba237bdc152c5b621a6d6955`, retaining only joint PFM for Chair
and Airplane. Packaging, portable runtime, data bounds, documentation and tests
were adapted for this release. Dependencies are installed separately under their
respective licenses. ShapeNet and external initialization weights are excluded;
this software license does not grant rights to those external assets.
