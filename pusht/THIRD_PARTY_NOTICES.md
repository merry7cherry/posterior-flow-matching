# Third-party source notices

This code is a focused derivative of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), upstream commit `5ba07ac6661db573af695b419a7947ecb704690f`. The original Columbia Artificial Intelligence and Robotics Lab MIT notice is retained in `LICENSE`. PFM additions use that license.

Inherited descendants retain the following upstream licenses:

- `diffusion_policy/gym_util/async_vector_env.py` adapts OpenAI Gym vector-environment code (MIT); see `licenses/gym-MIT.txt`.
- `diffusion_policy/env/pusht/pymunk_override.py` contains Pymunk drawing utilities and their original Victor Blomqvist MIT notice; see also `licenses/pymunk-MIT.txt`.
- `diffusion_policy/model/vision/crop_randomizer.py` and the observation encoder/configuration integration derive from or adapt robomimic utilities (MIT); see `licenses/robomimic-MIT.txt`.
- The EMA and cosine scheduler descend from Diffusion Policy's diffusers integration. The cosine formula is retained without a runtime diffusers dependency. See the diffusers Apache-2.0 text in `licenses/diffusers-Apache-2.0.txt`.
- The U-Net follows Diffusion Policy's adaptation of [Planning with Diffusion](https://github.com/jannerm/diffuser) (MIT; Phil Wang, Michael Janner and Yilun Du); its full notice is `licenses/diffuser-MIT.txt`.
- The PushT environment follows Diffusion Policy's adaptation of [Google Research IBC](https://github.com/google-research/ibc) (Apache-2.0); see `licenses/ibc-Apache-2.0.txt`.
- The `diffusion_policy` namespace identifies the retained upstream architecture and simulator; this package implements only PFM training and sampling.

Dependencies are installed separately and retain their own licenses. The official PushT dataset is downloaded separately from its publisher; this repository does not relicense or redistribute it. robomimic's unused EGL discovery dependency is excluded for this native pygame simulator.
