"""Native pointwise coordinate backbone from diffusion-point-cloud (MIT)."""
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from .common import ConcatSquashLinear

class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

    def hidden(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for layer in self.layers[:-1]:
            out = self.act(layer(ctx=ctx_emb, x=out))
        return out, ctx_emb

    def forward(self, x, beta, context):
        hidden, ctx_emb = self.hidden(x, beta, context)
        out = self.layers[-1](ctx=ctx_emb, x=hidden)

        if self.residual:
            return x + out
        else:
            return out
