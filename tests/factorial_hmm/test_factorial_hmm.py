import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

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

# def test_gaussian_arhmm_sample_is_consistent():
#     # sampled from ARHMM previously
#     true_states = np.array([[0, 2, 2],
#                             [0, 0, 0]], dtype=np.int32)
#     true_data = np.array([[[ 1.1885295 ,  0.55225945],
#                            [ 1.3390907 ,  0.8323249 ],
#                            [ 2.7013097 ,  1.469697  ]],
#                           [[-1.3038868 , -1.4337387 ],
#                            [-2.3088303 , -0.5458503 ],
#                            [ 0.1273387 , -1.296866  ]]], dtype=np.float32)

#     rng1, rng2 = jr.split(SEED, 2)
#     arhmm = GaussianARHMM(3, 2, 1, seed=rng1)
#     states, data = arhmm.sample(rng2, num_steps=3, num_samples=2)
#     assert np.all(true_states == states)
#     assert np.allclose(true_data, data)

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
