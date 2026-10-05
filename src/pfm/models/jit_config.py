from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JiTModelSpec:
    name: str
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    bottleneck_dim: int
    in_context_len: int
    in_context_start: int
    image_sizes: tuple[int, ...]


JIT_MODEL_SPECS: dict[str, JiTModelSpec] = {
    "JiT-B/16": JiTModelSpec("JiT-B/16", 16, 768, 12, 12, 128, 32, 4, (256,)),
    "JiT-L/16": JiTModelSpec("JiT-L/16", 16, 1024, 24, 16, 128, 32, 8, (256,)),
    "JiT-H/16": JiTModelSpec("JiT-H/16", 16, 1280, 32, 16, 256, 32, 10, (256,)),
}


def jit_spec_for(name: str, image_size: int) -> JiTModelSpec:
    try:
        spec = JIT_MODEL_SPECS[str(name)]
    except KeyError as error:
        raise ValueError(f"Unknown JiT model {name!r}; expected one of {tuple(JIT_MODEL_SPECS)}") from error
    if int(image_size) not in spec.image_sizes:
        raise ValueError(
            f"JiT model {spec.name} does not support image_size={int(image_size)}; "
            f"expected one of {spec.image_sizes}"
        )
    if int(image_size) % int(spec.patch_size) != 0:
        raise ValueError(f"image_size={image_size} must be divisible by patch_size={spec.patch_size}")
    return spec
