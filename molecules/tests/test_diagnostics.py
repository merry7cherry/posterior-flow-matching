
import numpy as np
import torch

from etflow.cli import diagnostics as d
from etflow.models import pfm_probability as prob


def test_projected_gaussian_calibration_and_affine_consistency():
    rng = np.random.default_rng(77)
    n = 7
    g = np.linspace(.3, 2., n)
    noise = rng.normal(size=(30000,n,3))*np.sqrt(g)[None,:,None]
    residual = noise-noise.mean(1,keepdims=True)
    marginal = (1-2/n)*g + g.sum()/n**2
    a = d.atom_statistics(residual, marginal, .2)
    assert abs(a['ratio'].mean()-1) < .01
    np.testing.assert_allclose(a['error_x']/a['trace_x'], a['ratio'])
    # This catches incorrectly using pre-projection scales or forgetting trace's 3.
    np.testing.assert_allclose(a['error_x'].mean(0), .04*3*marginal, rtol=.02)


def test_multiple_conformers_are_selected_reproducibly():
    a = d.pair_indices(20,8,4,42)
    np.testing.assert_array_equal(a,d.pair_indices(20,8,4,42))
    assert len(np.unique(a))==8 and len(a)==32
    np.testing.assert_array_equal(np.unique(a,return_counts=True)[1],4)
    assert len(d.pair_indices(2,8,4,42))==8


def test_equal_molecule_weighting_not_atom_pooling():
    values=np.array([1.,10.,10.,10.,10.])
    weights=np.array([1.,.25,.25,.25,.25])
    assert d.weighted_quantiles(values,weights,[.5])[0] < 10.


def test_variance_underestimation_detected():
    residual=np.ones((1,4,3))
    a=d.atom_statistics(residual,np.full((1,4),.25),.01)
    np.testing.assert_allclose(a['ratio'],4)


def test_summary_detects_centered_residual_errors():
    # Centering makes the pooled first moment zero even with a poor prediction.
    residual = torch.tensor([[2., 0., 0.], [-2., 0., 0.],
                             [0., 1., 0.], [0., -1., 0.]])
    batch = torch.zeros(4, dtype=torch.long)
    logvar = torch.zeros(4)
    marginal = prob.marginal_variance(logvar.exp(), batch).numpy()[None]
    config = dict(times=[.5], bins=1, bootstrap_repeats=32, seed=42)
    design = dict(inner_bin_edges=[[]], time_only_preprojection_variance=[1.])

    def summary(scale):
        error = scale * residual
        torch.testing.assert_close(error.mean(0), torch.zeros(3))
        record = {k: value[None] for k, value in
                  d.atom_statistics(error.numpy()[None], marginal, .5).items()}
        record.update(num_atoms=4, nll=prob.projected_nll(error, logvar, batch).numpy()[None],
                      bound_fraction=np.zeros((1, 1, 4)))
        return d.summarize([record], design, config)['timepoints'][0]

    original, worse = summary(1), summary(2)
    assert original['error_x']['mean'] > 0
    assert worse['error_x']['mean'] == 4 * original['error_x']['mean']
    assert worse['ratio']['mean'] == 4 * original['ratio']['mean']
    assert worse['trace_x'] == original['trace_x']
    assert worse['nll']['mean'] > original['nll']['mean']
    assert worse['baseline_nll']['mean'] > original['baseline_nll']['mean']
