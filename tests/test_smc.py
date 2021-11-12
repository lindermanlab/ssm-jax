"""
Test code for SMC filtering/smoothing for SSMs.
"""
import tensorflow_probability.substrates.jax as tfp
from jax.interpreters import xla
from jax import vmap
import jax
import jax.scipy.special as spsp
import jax.random as jr
import jax.numpy as np
import pytest

from ssm.hmm import GaussianHMM
from ssm.arhmm import GaussianARHMM
from ssm.distributions.glm import GaussianLinearRegression
from ssm.lds import GaussianLDS

from ssm.inference.smc import smc


LML_RTOL = 0.2  # Allowable deviation of ML ratio from 1.0.
LML_DISCARD = 1  # The number of high and low ratios that can be discarded

NUM_ROUNDS = 10
NUM_TRIALS = 5
NUM_TIMESTEPS = 200
LATENT_DIM = 3
EMISSIONS_DIM = 10
NUM_PARTICLES = 5000

NUM_TRIALS_SWEEP = range(1, 202, 100)
NUM_TIMESTEPS_SWEEP = range(10, 20011, 10000)
LATENT_DIM_SWEEP = range(2, 13, 2)
EMISSIONS_DIM_SWEEP = range(2, 13, 2)


@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()


def single_test(key, model_family, num_trials, num_timesteps, num_particles, latent_dim, emissions_dim):

    # Define the model.
    key, subkey = jr.split(key)
    model = model_family(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey)

    # Sample the data.
    key, subkey = jr.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Test against EM (which for the LDS is exact.
    em_posterior = vmap(model.infer_posterior)(data)
    em_log_marginal_likelihood = model.marginal_likelihood(data, posterior=em_posterior)

    # Close over a single SMC sweep.
    repeat_smc = lambda _k: smc(_k, model, data, proposal=None, num_particles=num_particles)
    vmapped = vmap(repeat_smc)

    # Run all the sweeps.
    key, subkey = jr.split(key)
    _, smc_log_marginal_likelihood, _, _ = vmapped(jr.split(subkey, num=NUM_ROUNDS))

    # Compute the expected LML using SMC.
    expected_smc_lml = spsp.logsumexp(smc_log_marginal_likelihood, axis=0) - len(smc_log_marginal_likelihood)

    # Compute the fractional difference.
    lml_diff = expected_smc_lml - em_log_marginal_likelihood
    ml_diff = np.exp(lml_diff)

    # Discard the outliers.
    for _ in range(LML_DISCARD):
        ml_diff = np.delete(ml_diff, ml_diff.argmin())
        ml_diff = np.delete(ml_diff, ml_diff.argmax())

    # Compute the mean.
    mml_diff = np.mean(ml_diff)

    # Test if this is within the tolerance.
    passed = (1.0 - LML_RTOL) < mml_diff < (1.0 + LML_RTOL)

    return passed


def test_smc_lds():

    closed_single_test = lambda _k, _trials, _timesteps, _particles, _latent_dim, _emissions_dim: \
        single_test(_k, GaussianLDS, _trials, _timesteps, _particles, _latent_dim, _emissions_dim)

    key = jr.PRNGKey(0)
    assert closed_single_test(key, NUM_TRIALS, NUM_TIMESTEPS, NUM_PARTICLES, LATENT_DIM, EMISSIONS_DIM)


    p = 0

test_smc_lds()
