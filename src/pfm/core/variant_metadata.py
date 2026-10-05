from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantCapabilities:
    predicts_logvar: bool
    supports_posterior_diagnostics: bool
    supports_generation_variance_maps: bool


@dataclass(frozen=True)
class MethodProfile:
    variant: str


DELTA_VARIANTS = frozenset({"pfm_delta"})


VARIANT_CAPABILITIES = {
    "pfm_full": VariantCapabilities(True, True, True),
    "pfm_delta": VariantCapabilities(True, True, True),
    "endpoint_fm": VariantCapabilities(False, False, False),
    "velocity_fm": VariantCapabilities(False, False, False),
}
SUPPORTED_VARIANTS = frozenset(VARIANT_CAPABILITIES)
CANONICAL_METHOD_PROFILES = SUPPORTED_VARIANTS


def variant_capabilities(variant: str) -> VariantCapabilities:
    try:
        return VARIANT_CAPABILITIES[str(variant)]
    except KeyError as exc:
        expected = tuple(sorted(VARIANT_CAPABILITIES))
        raise ValueError(f"Unknown variant {variant!r}; expected one of {expected}") from exc


def validate_base_model_config(variant: str, config) -> None:
    variant_capabilities(variant)
    if config.self_conditioning_mode != "none":
        raise ValueError(
            "self_conditioning_mode must be 'none'; self-conditioning runtime is retired. "
            "Use the historical code revision to reproduce self-conditioned checkpoints."
        )

def method_profile_name(variant: str) -> str:
    variant_capabilities(variant)
    return str(variant)


def parse_method_profile(value: object) -> MethodProfile:
    if not isinstance(value, str) or value not in CANONICAL_METHOD_PROFILES:
        raise ValueError(
            "method profile must be one of: "
            f"{sorted(CANONICAL_METHOD_PROFILES)}"
        )
    return MethodProfile(variant=value)


def requires_learned_variance(variant: str, *, feature: str) -> None:
    capabilities = variant_capabilities(variant)
    if not capabilities.predicts_logvar:
        raise ValueError(
            f"{feature} requires learned variance, but variant {variant!r} does not predict logvar"
        )
