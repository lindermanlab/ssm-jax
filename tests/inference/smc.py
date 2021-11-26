import pytest
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
import ssm.utils as utils
import ssm.inference.proposals as proposals


SEED = jr.PRNGKey(0)

# Allowable deviation of ML ratio from 1.0.
LML_RTOL = 0.1


@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()


def _single_smc_test(key, model, proposal, data, num_rounds, num_particles, ):

    # Test against EM (which for the LDS is exact.
    em_posterior = vmap(model.e_step)(data)
    em_log_marginal_likelihood = model.marginal_likelihood(data, posterior=em_posterior)

    # Close over a single SMC sweep.
    repeat_smc = lambda _k: smc(_k, model, data, proposal=proposal, num_particles=num_particles)
    vmapped = vmap(repeat_smc)

    # Run all the sweeps.
    key, subkey = jr.split(key)
    smc_posteriors = vmapped(jr.split(subkey, num=num_rounds))

    return em_log_marginal_likelihood, smc_posteriors


def test_smc_gaussian_lds_runs():
    """
    Verify that SMC still runs.
    """
    prop_fn = None
    num_rounds = 2
    num_trials = 3
    num_timesteps = 10
    num_particles = 15
    latent_dim = 4
    emissions_dim = 5

    key, subkey = jr.split(SEED)

    model_kwargs = {'dynamics_scale_tril': 0.001 * np.eye(latent_dim),
                    'dynamics_weights': np.eye(latent_dim),
                    'emission_scale_tril': 0.1 * np.eye(emissions_dim),}

    # Define the model.
    key, subkey = jr.split(key)
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, **model_kwargs)

    # Sample the data.
    key, subkey = jr.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Define the proposal to use.
    # TODO - find a better place to store this.
    from tests.inference._test_fivo import lds_define_proposal
    proposal_structure = 'RESQ'
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = lds_define_proposal(subkey, model, data, proposal_structure)
    prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)(proposal_params)

    # Run a single test.
    em_log_marginal_likelihood, smc_posteriors = _single_smc_test(key, model, prop_fn, data, num_rounds,
                                                                  num_particles, )

    smc_posteriors.sample(seed=key)

    # Check the shapes.
    assert smc_posteriors.log_normalizer.shape == (num_rounds, num_trials), \
        "Failed: Log normalizer shape: {} should be {}.".format(str(smc_posteriors.log_normalizer.shape),
                                                                str((num_rounds, num_trials)))

    assert smc_posteriors.batch_shape == (num_rounds, ), \
        "Failed: Batch shape: {} should be {}.".format(str(smc_posteriors.batch_shape),
                                                       str((num_rounds, )))

    assert smc_posteriors.event_shape == (num_trials, num_timesteps, num_particles, latent_dim), \
        "Failed: Event shape: {} should be {}.".format(str(smc_posteriors.event_shape),
                                                       str((num_trials, num_timesteps, num_particles, latent_dim)))

    try:
        smc_posteriors.sample(seed=subkey)
    except Exception as err:
        "Failed: Sampling from distribution: {}".format(err)

    assert smc_posteriors.sample(seed=subkey).shape == (num_rounds, num_trials, num_timesteps, latent_dim), \
        "Failed: Sample shape: {} should be {}.".format(str(smc_posteriors.event_shape),
                                                        str((num_rounds, num_trials, num_timesteps, latent_dim)))

    try:
        assert smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey))
    except Exception as err:
        print("Failed: Log prob evaluation: {}.".format(err))

    assert np.all(smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)) != 0.0), \
        "Failed: Log prob of particles was zero."

    assert np.all(np.isfinite(smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)))), \
        "Failed: Log prob of particles was infinite."

    assert smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)).shape == (num_rounds, num_trials), \
        "Failed: Log probability shape: {} should be {}.".format(str(smc_posteriors.log_normalizer.shape),
                                                                 str((num_rounds, num_trials)))

    return None


def test_smc_single_sweep_gaussian_lds():
    pass


@pytest.mark.slow
def test_smc_single_model_gaussian_lds():
    """
    Test the bootstrap particle filter implementation.
    """

    num_rounds = 5
    num_trials = 3
    num_timesteps = 100
    num_particles = 1000
    latent_dim = 3
    emissions_dim = 5

    key, subkey = jr.split(SEED)

    model_kwargs = {'dynamics_scale_tril': 0.1 * np.eye(latent_dim),
                    'emission_scale_tril': 0.5 * np.eye(emissions_dim)}

    # Define the model.
    key, subkey = jr.split(key)
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, **model_kwargs)

    # Sample the data.
    key, subkey = jr.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    assert _single_smc_test(key, model, None, data, num_rounds, num_particles, )


@pytest.mark.slow
def test_smc_multiple_models_gaussian_lds():
    pass


if __name__ == '__main__':
    test_smc_gaussian_lds_runs()
    test_smc_single_sweep_gaussian_lds()
    test_smc_single_model_gaussian_lds()
    test_smc_multiple_models_gaussian_lds()