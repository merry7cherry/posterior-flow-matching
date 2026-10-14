"""Plain PFM checkpoints: model, EMA, optimizer, configuration and epoch progress."""
from dataclasses import asdict
from pathlib import Path
import torch
from pfm.config import ImageConfig
from pfm.models.factory import build_model


def save_checkpoint(path, model, ema, optimizer, config, epoch, step, rng_states):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
                "rng_states": rng_states}, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ImageConfig(**checkpoint["config"])
    model = build_model(config, device)
    model.load_state_dict(checkpoint["ema"])
    return model.eval(), config
