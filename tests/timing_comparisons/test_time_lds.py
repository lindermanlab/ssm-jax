import pytest
from tensorflow_probability.substrates import jax as tfp
import jax.random as jr
import jax.numpy as np

from ssm.distributions.linreg import GaussianLinearRegression
from ssm.distributions.glm import PoissonGLM
from ssm.lds import GaussianLDS, PoissonLDS
from ssm.utils import random_rotation

import config

from jax.interpreters import xla


def create_random_lds(
    emission_dim=config.EMISSIONS_DIM,
    latent_dim=config.LATENT_DIM,
    rng=jr.PRNGKey(0),
    emissions="gaussian",
):

    if emissions == "gaussian":
        lds = GaussianLDS(
            num_latent_dims=latent_dim, num_emission_dims=emission_dim, seed=rng
        )

    elif emissions == "poisson":
        lds = PoissonLDS(
            num_latent_dims=latent_dim, num_emission_dims=emission_dim, seed=rng
        )

    return lds


def lds_fit_setup(
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
    true_lds = create_random_lds(emissions_dim, latent_dim, true_rng, emissions)
    states, data = true_lds.sample(sample_rng, num_timesteps, num_samples=num_trials)
    test_lds = create_random_lds(emissions_dim, latent_dim, test_rng, emissions)
    print("")  # for verbose pytest, this prevents tqdm from clobering pytest's layout
    return test_lds, data, num_iters, params


def lds_fit_em(lds, data, num_iters, params):
    lp, fit_model, posteriors = lds.fit(data, method="em", num_iters=num_iters, tol=-1)
    last_lp = lp[-1].block_until_ready()  # explicitly block until ready
    return lp, params


def lds_fit_laplace_em(lds, data, num_iters, params, rng=jr.PRNGKey(0)):
    lp, fit_model, posteriors = lds.fit(
        data, method="laplace_em", num_iters=num_iters, tol=-1, rng=rng
    )
    last_lp = lp[-1].block_until_ready()  # explicitly block until ready
    return lp, params


def run_time_test(benchmark, time_fn, setup_fn):
    lp, params = benchmark.pedantic(
        time_fn, setup=setup_fn, rounds=config.NUM_ROUNDS
    )
    benchmark.extra_info["params"] = params
    assert not np.any(np.isnan(lp))


#### Gaussian LDS EM TESTS
class TestGaussianLDSEM:
    @pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
    def test_lds_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (lds_fit_setup(num_trials=num_trials), {})
        run_time_test(benchmark, lds_fit_em, setup)

    @pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP)
    def test_lds_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (lds_fit_setup(num_timesteps=num_timesteps), {})
        run_time_test(benchmark, lds_fit_em, setup)

    @pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
    def test_lds_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (lds_fit_setup(latent_dim=latent_dim), {})
        run_time_test(benchmark, lds_fit_em, setup)

    @pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
    def test_lds_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (lds_fit_setup(emissions_dim=emissions_dim), {})
        run_time_test(benchmark, lds_fit_em, setup)


#### PLDS EM TESTS
class TestPoissonLDSLaplaceEM:
    @pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
    def test_lds_laplace_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (lds_fit_setup(num_trials=num_trials, emissions="poisson"), {})
        run_time_test(benchmark, lds_fit_laplace_em, setup)

    @pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP)
    def test_lds_laplace_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (
            lds_fit_setup(num_timesteps=num_timesteps, emissions="poisson"),
            {},
        )
        run_time_test(benchmark, lds_fit_laplace_em, setup)

    @pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
    def test_lds_laplace_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (lds_fit_setup(latent_dim=latent_dim, emissions="poisson"), {})
        run_time_test(benchmark, lds_fit_laplace_em, setup)

    @pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
    def test_lds_laplace_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (
            lds_fit_setup(emissions_dim=emissions_dim, emissions="poisson"),
            {},
        )
        run_time_test(benchmark, lds_fit_laplace_em, setup)
