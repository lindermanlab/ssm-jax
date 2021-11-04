import pytest
from tensorflow_probability.substrates import jax as tfp
import jax.random as jr
import jax.numpy as np

from ssm.distributions.linreg import GaussianLinearRegression
from ssm.distributions.glm import PoissonGLM
from ssm.lds import GaussianLDS, PoissonLDS
from ssm.utils import random_rotation

from config import NUM_ROUNDS

def create_random_lds(emission_dim=10, latent_dim=2, rng=jr.PRNGKey(0), emissions="gaussian"):

    if emissions == "gaussian":
        lds = GaussianLDS(num_latent_dims=latent_dim,
                          num_emission_dims=emission_dim,
                          seed=rng)

    elif emissions == "poisson":
        lds = PoissonLDS(num_latent_dims=latent_dim,
                         num_emission_dims=emission_dim,
                         seed=rng)

    return lds


def lds_fit_setup(num_trials=5, num_timesteps=100, latent_dim=3, emissions_dim=10, num_iters=100, emissions="gaussian"):
    rng = jr.PRNGKey(0)
    true_rng, sample_rng, test_rng = jr.split(rng, 3)
    true_lds = create_random_lds(emissions_dim, latent_dim, true_rng, emissions)
    states, data = true_lds.sample(sample_rng, num_timesteps, num_samples=num_trials)
    test_lds = create_random_lds(emissions_dim, latent_dim, test_rng, emissions)
    print("")  # for verbose pytest, this prevents tqdm from clobering pytest's layout
    return test_lds, data, num_iters


def lds_fit_em(lds, data, num_iters):
    lp, fit_model, posteriors = lds.fit(data, method="em", num_iters=num_iters, tol=-1)
    last_lp = lp[-1].block_until_ready  # explicitly block until ready
    return lp

def lds_fit_laplace_em(lds, data, num_iters, rng=jr.PRNGKey(0)):
    lp, fit_model, posteriors = lds.fit(data, method="laplace_em", num_iters=num_iters, tol=-1, rng=rng)
    last_lp = lp[-1].block_until_ready  # explicitly block until ready
    return lp


#### Gaussian LDS EM TESTS
class TestGaussianLDSEM:
    @pytest.mark.parametrize("num_trials", range(1, 202, 50))
    def test_lds_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (lds_fit_setup(num_trials=num_trials), {})
        lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))
        
    @pytest.mark.parametrize("num_timesteps", range(10, 20011, 10000))
    def test_lds_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (lds_fit_setup(num_timesteps=num_timesteps), {})
        lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("latent_dim", range(2, 13, 5))
    def test_lds_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (lds_fit_setup(latent_dim=latent_dim), {})
        lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("emissions_dim", range(2, 13, 5))
    def test_lds_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (lds_fit_setup(emissions_dim=emissions_dim), {})
        lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))


#### PLDS EM TESTS
class TestPoissonLDSLaplaceEM:
    @pytest.mark.parametrize("num_trials", range(1, 202, 50))
    def test_lds_laplace_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (lds_fit_setup(num_trials=num_trials, emissions="poisson"), {})
        lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))
        
    @pytest.mark.parametrize("num_timesteps", range(10, 1011, 250))
    def test_lds_laplace_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (lds_fit_setup(num_timesteps=num_timesteps, emissions="poisson"), {})
        lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("latent_dim", range(2, 13, 5))
    def test_lds_laplace_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (lds_fit_setup(latent_dim=latent_dim, emissions="poisson"), {})
        lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("emissions_dim", range(2, 13, 5))
    def test_lds_laplace_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (lds_fit_setup(emissions_dim=emissions_dim, emissions="poisson"), {})
        lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=NUM_ROUNDS)
        assert not np.any(np.isnan(lp))