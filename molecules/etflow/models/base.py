from lightning.pytorch import LightningModule


class BaseModel(LightningModule):
    def __init__(self, optimizer_type="AdamW", lr=1e-3, beta1=.95, beta2=.999,
                 weight_decay=0., min_lr=1e-8):
        super().__init__()
        if optimizer_type != "AdamW" or not 0 <= min_lr <= lr:
            raise ValueError("PFM uses AdamW with 0 <= min_lr <= lr")
        self.lr, self.min_lr = lr, min_lr
        self.opt_betas = (beta1, beta2)
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "val")
