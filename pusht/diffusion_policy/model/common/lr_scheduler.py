# Adapted for the PFM-only cosine path from Diffusion Policy/diffusers (Apache-2.0).
# See THIRD_PARTY_NOTICES.md and licenses/diffusers-Apache-2.0.txt.
"""Native cosine warmup schedule, without the unused diffusion dependency."""
import math
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    if name != 'cosine':
        raise ValueError('PFM uses cosine learning-rate decay')
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step-num_warmup_steps) / float(max(1, num_training_steps-num_warmup_steps))
        return max(0.0, 0.5*(1.0+math.cos(math.pi*progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
