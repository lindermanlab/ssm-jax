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
    
def test_poisson_lds_sample():
    rng1, rng2 = jr.split(SEED, 2)
    lds = PoissonLDS(3, 5, seed=rng1)
    states, data = lds.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10, 3)
    assert data.shape == (32, 10, 5)
    
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
