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
    true_states_var0 = np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                 [2, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.int32)
    true_states_var1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 1, 1, 1, 0, 0, 2, 2, 2]], dtype=np.int32)
    true_data = np.array([[ 0.6167348 ,  0.64424795,  0.61341816,  0.591818  ,
                            0.6095528 ,  0.62132317,  0.61306655,  0.6258184 ,
                            0.6209738 ,  0.6320318 ],
                          [ 0.61753726,  0.08578929,  0.09964492,  0.08741929,
                            0.06252673, -0.29311082, -0.28715354,  0.4724555 ,
                            -2.3517041 , -2.3381226 ]],
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
