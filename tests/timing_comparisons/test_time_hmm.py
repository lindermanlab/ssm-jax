import tensorflow_probability.substrates.jax as tfp
from jax.interpreters import xla
import jax.random as jr
import jax.numpy as np
import pytest

from ssm.hmm import GaussianHMM
from ssm.arhmm import GaussianARHMM
from ssm.distributions.glm import GaussianLinearRegression

import config


def make_random_hmm(
    emissions_dim=config.EMISSIONS_DIM,
    latent_dim=config.LATENT_DIM,
    rng=jr.PRNGKey(0),
    emissions="gaussian",
):
    num_states = latent_dim

    if emissions == "gaussian":
        emission_means = 3 * jr.normal(rng, shape=(latent_dim, emissions_dim))
        emission_covariances = np.tile(np.eye(emissions_dim), (num_states, 1, 1))
        return GaussianHMM(
            num_states,
            emission_means=emission_means,
            emission_covariances=emission_covariances,
        )

    elif emissions == "autoregressive":
        emission_weights = np.tile(0.99 * np.eye(emissions_dim), (num_states, 1, 1))
        emission_biases = 0.01 * jr.normal(rng, (num_states, emissions_dim))
        emission_covariances = np.tile(np.eye(emissions_dim), (num_states, 1, 1))
        return GaussianARHMM(
            num_states,
            emission_weights=emission_weights,
            emission_biases=emission_biases,
            emission_covariances=emission_covariances,
        )


def hmm_fit_em_setup(
    num_trials=config.NUM_TRIALS,
    num_timesteps=config.NUM_TIMESTEPS,
    latent_dim=config.LATENT_DIM,
    emissions_dim=config.EMISSIONS_DIM,
    num_iters=config.NUM_ITERS,
    emissions="gaussian",
):
    rng = jr.PRNGKey(0)
    params = locals()
    true_rng, sample_rng, test_rng = jr.split(rng, 3)
    true_hmm = make_random_hmm(emissions_dim, latent_dim, true_rng, emissions=emissions)
    states, data = true_hmm.sample(sample_rng, num_timesteps, num_samples=num_trials)
    test_hmm = make_random_hmm(emissions_dim, latent_dim, test_rng, emissions=emissions)
    print("")  # for verbose pytest, this prevents tqdm from clobering pytest's layout
    return test_hmm, data, num_iters, params


def hmm_fit_em(hmm, data, num_iters, params):
    lp, fit_model, posteriors = hmm.fit(data, method="em", num_iters=num_iters, tol=-1)
    last_lp = lp[-1].block_until_ready()  # explicitly block until ready
    return lp, params


@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()


def run_time_test(benchmark, time_fn, setup_fn):
    lp, params = benchmark.pedantic(
        time_fn, setup=setup_fn, rounds=config.NUM_ROUNDS
    )
    benchmark.extra_info["params"] = params
    assert not np.any(np.isnan(lp))


#### TEST GAUSSIAN HMM
class TestGaussianHMM:
    @pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
    def test_hmm_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (hmm_fit_em_setup(num_trials=num_trials), {})
        run_time_test(benchmark, hmm_fit_em, setup)

    @pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP)
    def test_hmm_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (hmm_fit_em_setup(num_timesteps=num_timesteps), {})
        run_time_test(benchmark, hmm_fit_em, setup)

    @pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
    def test_hmm_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (hmm_fit_em_setup(latent_dim=latent_dim), {})
        run_time_test(benchmark, hmm_fit_em, setup)

    @pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
    def test_hmm_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (hmm_fit_em_setup(emissions_dim=emissions_dim), {})
        run_time_test(benchmark, hmm_fit_em, setup)


#### TEST GAUSSIAN ARHMM
class TestGaussianARHMM:
    @pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
    def test_arhmm_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (
            hmm_fit_em_setup(num_trials=num_trials, emissions="autoregressive"),
            {},
        )
        run_time_test(benchmark, hmm_fit_em, setup)

    @pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP)
    def test_arhmm_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (
            hmm_fit_em_setup(num_timesteps=num_timesteps, emissions="autoregressive"),
            {},
        )
        run_time_test(benchmark, hmm_fit_em, setup)

    @pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
    def test_arhmm_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (
            hmm_fit_em_setup(latent_dim=latent_dim, emissions="autoregressive"),
            {},
        )
        run_time_test(benchmark, hmm_fit_em, setup)

    @pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
    def test_arhmm_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (
            hmm_fit_em_setup(emissions_dim=emissions_dim, emissions="autoregressive"),
            {},
        )
        run_time_test(benchmark, hmm_fit_em, setup)
