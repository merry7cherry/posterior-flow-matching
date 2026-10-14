"""Train PFM, evaluate native PushT rollouts, or inspect reference posteriors."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusion_policy.diagnostics import reference_diagnostics


def setup_cpu(seed=321):
    torch.set_num_threads(int(os.environ.get('PFM_CPU_THREADS', '2')))
    torch.manual_seed(seed)
    np.random.seed(seed)


@hydra.main(version_base=None, config_path='config', config_name='pusht')
def train(cfg):
    setup_cpu(cfg.training.seed)
    hydra.utils.get_class(cfg._target_)(cfg).run()


def load_policy(checkpoint, device):
    # Training checkpoints contain optimizer and RNG state; load trusted files.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(payload['config'])
    policy = hydra.utils.instantiate(cfg.policy)
    weights = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload[weights])
    return policy.to(device).eval(), cfg, weights


def checkpoint_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=321)
    return parser


def evaluate():
    parser = checkpoint_parser('Native PushT rollout scores and video')
    parser.add_argument('--solver', choices=['ode', 'sde'], default='ode')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--diffusion-scale', type=float, default=None)
    parser.add_argument('--n-test', type=int, default=50)
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=300)
    args = parser.parse_args()
    setup_cpu(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    policy, cfg, weights = load_policy(args.checkpoint, args.device)
    scale = args.diffusion_scale if args.diffusion_scale is not None else (.1 if args.solver == 'sde' else 0.)
    policy.sampling.update(solver=args.solver, steps=args.steps, diffusion_scale=scale, seed=args.seed)
    cfg.task.env_runner.update(n_train=0, n_train_vis=0, n_test=args.n_test,
                               n_test_vis=min(1, args.n_test), n_envs=args.n_envs,
                               max_steps=args.max_steps)
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(args.output))
    try:
        result = runner.run(policy)
    finally:
        runner.env.close()
    result.update(sampling=policy.sampling, device=args.device,
                  checkpoint=str(args.checkpoint), weights=weights)
    (args.output/'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def diagnose():
    parser = checkpoint_parser('Reference posterior diagnostics without simulator rollouts')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    setup_cpu(args.seed)
    policy, cfg, _ = load_policy(args.checkpoint, args.device)
    cfg.task.dataset.zarr_path = str(args.data)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    records = reference_diagnostics(policy, dataset, batches=args.batches, batch_size=args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'diagnostics.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))
