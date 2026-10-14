"""Joint point-cloud PFM training."""
import argparse
import json
import math
from pathlib import Path
import torch
from .common import get_linear_scheduler
from .generator import PFMGenerator
from .pfm import delta_loss, network_context
from .dataset import ShapeNetCore
from .runtime import load_config, seed_all, rng_state, restore_rng


BASE_STATS = ('loss', 'mse', 'variance', 'grad_norm', 'pfm_loss', 'latent_kl')

def train_step(model, x1, optimizer, scheduler, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z, kl = model.encode_with_kl(x1)
    x0 = torch.randn_like(x1)
    eps = config['eps_t']
    t = eps + (1 - 2 * eps) * torch.rand(len(x1), device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    with network_context(x1.device, config['precision']):
        velocity, ell = model.net(xt, t, z)
    pfm_loss = delta_loss(velocity, ell, x0, x1)
    loss = pfm_loss + config.get('kl_weight', 0.0) * kl
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.training_parameters(), config['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    values = (loss.detach(), (x1-x0-velocity).square().mean().detach(),
              ell.exp().mean().detach(), grad.detach(), pfm_loss.detach(), kl.detach())
    return torch.stack(values)


@torch.no_grad()
def validate(model, points, config, device):
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        model.eval()
        x1 = points[:32].to(device)
        z = model.encode(x1)
        x0 = torch.randn_like(x1)
        t = 0.001 + 0.998 * torch.rand(len(x1), device=device)
        xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
        with network_context(device, config['precision']):
            velocity, ell = model.net(xt, t, z)
        return {'loss': delta_loss(velocity, ell, x0, x1).item(),
                'variance': ell.exp().mean().item()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--condition', help='Category encoder/prior initialization for a new run')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='Checkpoint file; no original run directory is needed')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if not args.resume and (not args.config or not args.condition):
        parser.error('a new run needs --config and --condition')
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    config = checkpoint['config'] if checkpoint else load_config(args.config)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
        torch.backends.cudnn.allow_tf32 = config['allow_tf32']
    seed_all(config['seed'])
    model = (PFMGenerator.from_checkpoint(checkpoint) if checkpoint else
             PFMGenerator.from_official(args.condition, config['logvar_min'], config['logvar_max'])).to(device)
    optimizer = torch.optim.Adam(model.training_parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_linear_scheduler(optimizer, config['sched_start'], config['sched_end'], config['lr'], config['end_lr'])
    limit = checkpoint.get('limit_train') if checkpoint else args.limit_train
    dataset = ShapeNetCore(args.dataset, [config['category']], 'train', config['scale_mode'], limit=limit)
    points = torch.stack([item['pointcloud'] for item in dataset])
    # Native ordered batches retain the final partial batch. PointNet BatchNorm
    # needs at least two shapes, so choose a compatible batch size/subset.
    batches = math.ceil(len(points) / config['batch_size'])
    start = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start = checkpoint['iteration']
        restore_rng(checkpoint['rng'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    end = min(config['max_iters'], args.stop_after or config['max_iters'])
    accumulator = torch.zeros(len(BASE_STATS), device=device)
    count = 0
    with (output / 'train.jsonl').open('a' if checkpoint else 'w') as log:
        for iteration in range(start + 1, end + 1):
            offset = ((iteration - 1) % batches) * config['batch_size']
            x1 = points[offset:offset + config['batch_size']].to(device)
            accumulator += train_step(model, x1, optimizer, scheduler, config)
            count += 1
            if iteration % config['log_every'] == 0 or iteration == end:
                record = dict(zip(BASE_STATS, (accumulator / count).cpu().tolist()))
                record.update(iteration=iteration, lr=optimizer.param_groups[0]['lr'])
                line = json.dumps(record)
                print(line, flush=True)
                log.write(line + '\n'); log.flush()
                accumulator.zero_(); count = 0
            if iteration % config['validate_every'] == 0 or iteration == end:
                print(json.dumps({'training_probe': validate(model, points, config, device), 'iteration': iteration}))
            if iteration % config['checkpoint_every'] == 0 or iteration == end:
                state = {'iteration': iteration, 'config': config, 'native_args': model.native_args,
                         'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'limit_train': limit}
                torch.save(state, output / 'latest.pt')
                if iteration == config['max_iters']:
                    torch.save(state, output / 'final.pt')


if __name__ == '__main__':
    main()
