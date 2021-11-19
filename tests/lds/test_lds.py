import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

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
                             [ 2.143722  , -1.0683461 , -1.1490295 ],
                             [ 2.21696   , -1.1373025 , -0.9108291 ]],
                            [[ 0.17471106, -0.05582879, -1.9695592 ],
                             [ 0.24238327, -0.34164798, -1.9394323 ],
                             [ 0.31522033, -0.63146836, -1.8712306 ]]], dtype=np.float32)
    true_data = np.array([[[ 4.185537  , -0.3276755 ],
                           [ 3.05506   , -0.72648746],
                           [ 4.175408  , -0.05829722]],
                          [[ 0.12157202, -1.9061788 ],
                           [-0.56306684, -1.0952222 ],
                           [ 1.3611842 , -2.24853   ]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = GaussianLDS(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.allclose(true_states, states)
    assert np.allclose(true_data, data)
    
def test_poisson_lds_sample():
    rng1, rng2 = jr.split(SEED, 2)
    lds = PoissonLDS(3, 5, seed=rng1)
    states, data = lds.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10, 3)
    assert data.shape == (32, 10, 5)
    
def test_poisson_lds_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[[ 2.0503902 , -0.98077524, -1.389261  ],
                             [ 2.143722  , -1.0683461 , -1.1490295 ],
                             [ 2.21696   , -1.1373025 , -0.9108291 ]],
                            [[ 0.17471106, -0.05582879, -1.9695592 ],
                             [ 0.24238327, -0.34164798, -1.9394323 ],
                             [ 0.31522033, -0.63146836, -1.8712306 ]]], dtype=np.float32)
    true_data = np.array([[[17.,  0.],
                           [12.,  0.],
                           [15.,  0.]],
                          [[ 4.,  2.],
                           [ 1.,  0.],
                           [ 5.,  1.]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = PoissonLDS(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.allclose(true_states, states)
    assert np.allclose(true_data, data)
    
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
