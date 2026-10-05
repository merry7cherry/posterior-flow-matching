import pytest
import torch

from pfm.probability.diagonal_gaussian import (
    constant_free_nll,
    constant_free_nll_per_image,
    variance_from_logvar,
)


def test_variance_from_logvar_is_the_diagonal_velocity_variance():
    logvar = torch.tensor([[[[0.0, 1.0]]]])

    assert torch.allclose(variance_from_logvar(logvar), torch.exp(logvar))


def test_constant_free_nll_has_one_loss_representation():
    residual = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    logvar = torch.log(torch.tensor([[[[2.0, 4.0]]], [[[8.0, 16.0]]]]))
    expected_per_image = 0.5 * (
        torch.exp(-logvar) * residual.square() + logvar
    ).flatten(1).mean(dim=1)

    per_image = constant_free_nll_per_image(residual, logvar)
    loss = constant_free_nll(residual, logvar)

    assert torch.allclose(per_image, expected_per_image)
    assert loss == pytest.approx(float(expected_per_image.mean()))


def test_diagonal_gaussian_probability_math_is_fp32_for_bfloat16_inputs():
    residual = torch.tensor([[[[1.25, -0.75]]]], dtype=torch.bfloat16)
    logvar = torch.tensor([[[[0.5, -0.5]]]], dtype=torch.bfloat16)
    expected_variance = torch.exp(logvar.float())
    expected_nll = 0.5 * (
        torch.exp(-logvar.float()) * residual.float().square() + logvar.float()
    ).flatten(1).mean(dim=1)

    variance = variance_from_logvar(logvar)
    per_image = constant_free_nll_per_image(residual, logvar)

    assert variance.dtype == torch.float32
    assert per_image.dtype == torch.float32
    assert torch.allclose(variance, expected_variance)
    assert torch.allclose(per_image, expected_nll)
