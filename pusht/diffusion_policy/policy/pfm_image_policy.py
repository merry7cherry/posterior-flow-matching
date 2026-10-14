"""Native two-frame PushT policy with joint PFM mean and variance heads."""
import math
import torch
from torch import nn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision import crop_randomizer as dmvc
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.pfm import delta_loss, log_variance, sample_delta


class PFMImagePolicy(BaseImagePolicy):
    def __init__(self, shape_meta, horizon, n_action_steps, n_obs_steps,
                 crop_shape, diffusion_step_embed_dim, down_dims, kernel_size,
                 n_groups, cond_predict_scale, obs_encoder_group_norm,
                 eval_fixed_crop, logvar_min, logvar_max, logvar_init, eps_t,
                 noise_scale, time_embed_scale, mixed_precision, sampling):
        super().__init__()
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config()

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                )
            )


        self.obs_encoder = obs_encoder
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.horizon, self.action_dim = horizon, action_dim
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        self.model = ConditionalUnet1D(
            input_dim=action_dim, global_cond_dim=self.obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims,
            kernel_size=kernel_size, n_groups=n_groups,
            cond_predict_scale=cond_predict_scale)
        self.normalizer = LinearNormalizer()
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        self.eps_t, self.noise_scale = eps_t, noise_scale
        self.time_embed_scale, self.mixed_precision = time_embed_scale, mixed_precision
        self.sampling = dict(sampling)
        nn.init.zeros_(self.model.variance_head.weight)
        prob = (logvar_init-logvar_min)/(logvar_max-logvar_min)
        nn.init.constant_(self.model.variance_head.bias, math.log(prob/(1-prob)))
        self.loss_stats, self.sample_stats = {}, {}
        self.sample_trace = None
        self._sample_generators = None

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == 'cuda' and self.mixed_precision == 'bf16')

    def encode_observation(self, obs):
        """Normalize and encode the two conditioning frames once per action plan."""
        normalized = self.normalizer.normalize(obs)
        frames = dict_apply(normalized, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        with self._autocast():
            features = self.obs_encoder(frames)
        return features.reshape(features.shape[0] // self.n_obs_steps, -1)

    def _field(self, x, t, condition):
        with self._autocast():
            return self.model(x, t.float() * self.time_embed_scale, global_cond=condition)

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def clip_gradients(self, max_norm):
        norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm, error_if_nonfinite=True)
        return {'grad_norm': norm, 'grad_clip_scale': (max_norm / (norm + 1e-6)).clamp(max=1)}

    def compute_loss(self, batch):
        condition = self.encode_observation(batch['obs'])
        x1 = self.normalizer['action'].normalize(batch['action']).float()
        x0 = self.noise_scale * torch.randn_like(x1)
        t = self.eps_t + (1-2*self.eps_t) * torch.rand(len(x1), device=x1.device)
        xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
        velocity, raw = self._field(xt, t, condition)
        target = x1-x0
        ell = log_variance(raw, self.logvar_min, self.logvar_max)
        residual = (target.float()-velocity.float()).square()
        loss = delta_loss(velocity, raw, target, self.logvar_min, self.logvar_max)
        with torch.no_grad():
            self.loss_stats = {
                'loss_total': loss.detach(), 'velocity_mse': residual.mean(),
                'velocity_variance': ell.exp().mean(), 'logvar_mean': ell.mean(),
                'logvar_lower_saturation': (raw.float().sigmoid() < .01).float().mean(),
                'logvar_upper_saturation': (raw.float().sigmoid() > .99).float().mean()}
        return loss

    def reset(self):
        seed = int(self.sampling['seed'])
        self._sample_generators = [torch.Generator(device=self.device).manual_seed(seed+i)
                                   for i in range(3)]

    @torch.no_grad()
    def predict_action(self, obs_dict):
        condition = self.encode_observation(obs_dict)
        normalized = self.sample_actions(condition)
        action_pred = self.normalizer['action'].unnormalize(normalized)
        start = self.n_obs_steps - 1
        return {'action': action_pred[:, start:start+self.n_action_steps],
                'action_pred': action_pred}

    def sample_actions(self, condition):
        if self._sample_generators is None:
            self.reset()
        init_gen, brownian_gen, probe_gen = self._sample_generators
        initial = self.noise_scale * torch.randn(
            (len(condition), self.horizon, self.action_dim), device=self.device, generator=init_gen)
        condition = condition.detach()
        with torch.autocast(device_type=self.device.type, enabled=False):
            result, self.sample_stats = sample_delta(
                lambda x, t: self._field(x, t, condition), initial,
                steps=int(self.sampling['steps']), terminal_time=float(self.sampling['terminal_time']),
                solver=self.sampling['solver'], diffusion_scale=float(self.sampling['diffusion_scale']),
                probes=int(self.sampling['derivative_probes']), noise_scale=self.noise_scale,
                low=self.logvar_min, high=self.logvar_max,
                brownian_generator=brownian_gen, probe_generator=probe_gen,
                exact=self.sampling.get('exact_derivatives', False), trace=self.sample_trace)
        return result
