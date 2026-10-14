from torch import nn


class PosteriorHeads(nn.Module):
    """The two PFM heads share a backbone; the config names the mean coordinates."""
    def __init__(self, *, prediction_head_factory, logvar_init_bias):
        super().__init__()
        self.mean = prediction_head_factory(0.0)
        self.logvar = prediction_head_factory(logvar_init_bias)

    def forward(self, features, condition):
        return {"mean": self.mean(features, condition),
                "logvar": self.logvar(features, condition)}
