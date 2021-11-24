import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

import numpy as onp

from ssm.lds import GaussianLDS, PoissonLDS

SEED = jr.PRNGKey(0)

@jit
def identity(x):
    return x 

#### TESTS

def test_gaussian_lds_jit():
    lds = GaussianLDS(3, 5, seed=SEED)
    identity(lds)
    
def test_poisson_lds_jit():
    lds = PoissonLDS(3, 5, seed=SEED)
    identity(lds)
    
def test_gaussian_lds_sample():
    rng1, rng2 = jr.split(SEED, 2)
    lds = GaussianLDS(3, 5, seed=rng1)
    states, data = lds.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10, 3)
    assert data.shape == (32, 10, 5)
    
def test_gaussian_lds_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[[ 2.0503902 , -0.98077524, -1.389261  ],
                             [ 2.1278157 , -1.0894692 , -1.1714737 ],
                             [ 2.2172518 , -1.1573169 , -0.93505853]],
                            [[ 0.17471106, -0.05582879, -1.9695592 ],
                             [ 0.24917215, -0.34881595, -1.9439837 ],
                             [ 0.33026809, -0.6336352 , -1.8497025 ]]], dtype=np.float32)
    true_data = np.array([[[ 1.9926789 , -0.42406225],
                           [ 2.0938962 , -0.6344335 ],
                           [ 2.364299  , -0.9663689 ]],
                          [[ 0.984763  , -0.40625238],
                           [-1.1792403 , -2.729756  ],
                           [ 0.37870926,  0.0921855 ]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = GaussianLDS(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    onp.testing.assert_allclose(true_states, states, atol=1e-5)
    onp.testing.assert_allclose(true_data, data, atol=1e-5)
    
def test_poisson_lds_sample():
    rng1, rng2 = jr.split(SEED, 2)
    lds = PoissonLDS(3, 5, seed=rng1)
    states, data = lds.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10, 3)
    assert data.shape == (32, 10, 5)
    
def test_poisson_lds_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[[ 2.0503902 , -0.98077524, -1.389261  ],
                             [ 2.1278157 , -1.0894692 , -1.1714737 ],
                             [ 2.2172518 , -1.1573169 , -0.93505853]],
                            [[ 0.17471106, -0.05582879, -1.9695592 ],
                             [ 0.24917215, -0.34881595, -1.9439837 ],
                             [ 0.33026809, -0.6336352 , -1.8497025 ]]], dtype=np.float32)
    true_data = np.array([[[20.,  3.],
                           [13.,  0.],
                           [14.,  0.]],
                          [[ 5.,  1.],
                           [ 2.,  0.],
                           [ 4.,  1.]]], dtype=np.float32)
    lds = PoissonLDS(3, 2, seed=rng1)
    states, data = lds.sample(rng2, num_steps=3, num_samples=2)
    onp.testing.assert_allclose(true_states, states, atol=1e-5)
    onp.testing.assert_allclose(true_data, data, atol=1e-5)
    
def test_gaussian_lds_em_fit():
    rng1, rng2, rng3 = jr.split(SEED, 3)
    true_lds = GaussianLDS(3, 5, seed=rng1)
    states, data = true_lds.sample(rng2, num_steps=100, num_samples=3)
    test_lds = GaussianLDS(3, 5, seed=rng3)
    
    # fit with no early stopping 
    lp, fitted_hmm, posteriors = test_lds.fit(data, num_iters=100, tol=-1)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)
    
def test_poisson_lds_em_fit():
    rng1, rng2, rng3, rng4 = jr.split(SEED, 4)
    true_lds = PoissonLDS(3, 5, seed=rng1)
    states, data = true_lds.sample(rng2, num_steps=100, num_samples=3)
    test_lds = PoissonLDS(3, 5, seed=rng3)
    
    # fit with no early stopping 
    lp, fitted_hmm, posteriors = test_lds.fit(data, 
                                              num_iters=100,
                                              tol=-1,
                                              method="laplace_em",
                                              rng=rng4)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)
