
import numpy as np

from etflow.cli import diagnostics as d


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
