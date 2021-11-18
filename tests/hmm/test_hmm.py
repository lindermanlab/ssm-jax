import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

from ssm.hmm import GaussianHMM, PoissonHMM, BernoulliHMM

SEED = jr.PRNGKey(0)

@jit
def identity(x):
    return x 

#### TESTS

def test_gaussian_hmm_jit():
    hmm = GaussianHMM(3, 5, seed=SEED)
    identity(hmm)
    
def test_poisson_hmm_jit():
    hmm = PoissonHMM(3, 5, seed=SEED)
    identity(hmm)
    
def test_bernoulli_hmm_jit():
    hmm = BernoulliHMM(3, 5, seed=SEED)
    identity(hmm)
    
def test_gaussian_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = GaussianHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_poisson_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = PoissonHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_bernoulli_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = BernoulliHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_gaussian_hmm_em_fit():
    rng1, rng2, rng3 = jr.split(SEED, 3)
    true_hmm = GaussianHMM(3, 5, seed=rng1)
    states, data = true_hmm.sample(rng2, num_steps=100, num_samples=3)
    test_hmm = GaussianHMM(3, 5, seed=rng3)
    
    # fit with no early stopping 
    lp, fitted_hmm, posteriors = test_hmm.fit(data, num_iters=100, tol=-1)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)
    

def test_poisson_hmm_em_fit():
    rng1, rng2, rng3 = jr.split(SEED, 3)
    true_hmm = PoissonHMM(3, 5, seed=rng1)
    states, data = true_hmm.sample(rng2, num_steps=100, num_samples=3)
    test_hmm = PoissonHMM(3, 5, seed=rng3)
    
    # fit with no early stopping 
    lp, fitted_hmm, posteriors = test_hmm.fit(data, num_iters=100, tol=-1)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)
    

def test_bernoulli_hmm_em_fit():
    rng1, rng2, rng3 = jr.split(SEED, 3)
    true_hmm = BernoulliHMM(3, 5, seed=rng1)
    states, data = true_hmm.sample(rng2, num_steps=100, num_samples=3)
    test_hmm = BernoulliHMM(3, 5, seed=rng3)
    
    # fit with no early stopping 
    lp, fitted_hmm, posteriors = test_hmm.fit(data, num_iters=100, tol=-1)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)

