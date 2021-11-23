import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

import numpy as onp

from ssm.factorial_hmm import NormalFactorialHMM

SEED = jr.PRNGKey(0)

@jit
def identity(x):
    return x

#### TESTS

def test_normal_factorial_hmm_jit():

    fhmm = NormalFactorialHMM(num_states=(3, 4), seed=SEED)
    identity(fhmm)

def test_normal_factorial_hmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    fhmm = NormalFactorialHMM(num_states=(3, 4), seed=SEED)
    states, data = fhmm.sample(rng2, num_steps=10, num_samples=32)
    assert len(states) == 2
    assert states[0].shape == (32, 10)
    assert states[0].min() >= 0 and states[0].max() < 3
    assert states[1].shape == (32, 10)
    assert states[1].min() >= 0 and states[0].max() < 4
    assert data.shape == (32, 10)

def test_normal_factorial_hmm_sample_is_consistent():
    # sampled from FHMM previously
    true_states_var0 = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [2, 2, 2, 2, 2, 2, 2, 0, 0, 0]], dtype=np.int32)
    true_states_var1 = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 3, 3, 3, 3, 3, 3, 0, 0]], dtype=np.int32)
    true_data = np.array([[ 0.61314344, -0.28376377, -0.271816  ,  0.08673307,
                            0.08069246,  0.08510165,  0.08053012,  0.1039151 ,
                            0.08183516,  0.07629083],
                          [ 0.61972284,  0.6279197 ,  2.7789102 ,  2.794995  ,
                            2.7683432 ,  2.779642  ,  2.7876885 ,  1.8594146 ,
                            -0.29416072, -0.27387783]], 
                         dtype=np.float32)

    rng1, rng2 = jr.split(SEED, 2)
    fhmm = NormalFactorialHMM(num_states=(3, 4), seed=rng1)
    states, data = fhmm.sample(rng2, num_steps=10, num_samples=2)
    assert np.array_equal(states[0], true_states_var0)
    assert np.array_equal(states[1], true_states_var1)
    assert np.allclose(true_data, data, atol=1e-5)

def test_normal_factorial_hmm_em_fit():
    rng1, rng2, rng3 = jr.split(SEED, 3)
    true_fhmm = NormalFactorialHMM(num_states=(3, 4), seed=rng1)
    states, data = true_fhmm.sample(rng2, num_steps=100, num_samples=3)
    test_fhmm = NormalFactorialHMM(num_states=(3, 4), seed=rng3)

    # fit with no early stopping
    lp, fitted_fhmm, posteriors = test_fhmm.fit(data, num_iters=100, tol=-1)

    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3, 4)
