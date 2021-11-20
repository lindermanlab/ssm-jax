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
    
def test_gaussian_hmm_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[0, 2, 0],
                            [0, 0, 2]], dtype=np.int32)
    true_data = np.array([[[-0.29138386,  0.29394504],
                           [ 0.4951557 ,  1.1221184 ],
                           [ 0.19978702, -0.15766774]],
                          [[ 0.27224892, -0.09574001],
                           [-1.8726401 , -2.213043  ],
                           [ 0.350825  ,  1.7823567 ]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = GaussianHMM(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data, atol=1e-5)
    
def test_poisson_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = PoissonHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_poisson_hmm_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[0, 2, 0],
                            [0, 0, 2]], dtype=np.int32)
    true_data = np.array([[[2., 5.],
                           [3., 7.],
                           [2., 2.]],
                          [[2., 1.],
                           [1., 2.],
                           [4., 3.]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = PoissonHMM(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data, atol=1e-5)
    
def test_bernoulli_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = BernoulliHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_bernoulli_hmm_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[0, 2, 0],
                            [0, 0, 2]], dtype=np.int32)
    true_data = np.array([[[1, 0],
                           [1, 1],
                           [1, 0]],
                          [[1, 0],
                           [1, 1],
                           [1, 0]]], dtype=np.int32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = BernoulliHMM(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data, atol=1e-5)
    
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

