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
    
    true_states = np.array([[0, 2, 2],
                            [0, 0, 0]], dtype=np.int32)
    true_data = np.array([[[ 1.901474 ,  0.3903318],
                           [ 1.4439511,  1.0113559],
                           [ 2.636291 ,  1.7115304]],
                          [[-0.5909422, -1.5956663],
                           [-1.248173 , -0.5830741],
                           [ 0.7204445, -1.533948 ]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = GaussianHMM(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data)
    
def test_poisson_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = PoissonHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_poisson_hmm_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[0, 2, 2],
                            [0, 0, 0]], dtype=np.int32)
    true_data = np.array([[[2., 3.],
                           [1., 5.],
                           [3., 4.]],
                          [[2., 4.],
                           [0., 5.],
                           [4., 4.]]], dtype=np.float32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = PoissonHMM(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data)
    
def test_bernoulli_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    hmm = BernoulliHMM(3, 5, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_bernoulli_hmm_sample_is_consistent():
    rng1, rng2 = jr.split(SEED, 2)
    
    true_states = np.array([[0, 2, 2],
                            [0, 0, 0]], dtype=np.int32)
    true_data = np.array([[[1, 0],
                           [1, 1],
                           [0, 0]],
                          [[1, 1],
                           [1, 1],
                           [1, 1]]], dtype=np.int32)
    rng1, rng2 = jr.split(SEED, 2)
    hmm = BernoulliHMM(3, 2, seed=rng1)
    states, data = hmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data)
    
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

