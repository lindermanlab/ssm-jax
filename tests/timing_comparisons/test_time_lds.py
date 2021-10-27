import pytest
from tensorflow_probability.substrates import jax as tfp
import jax.random as jr
import jax.numpy as np

from ssm.distributions.linreg import GaussianLinearRegression
from ssm.distributions.glm import PoissonGLM
from ssm.models.lds import GaussianLDS, LDS
from ssm.plots import plot_dynamics_2d
from ssm.utils import random_rotation


def create_random_lds(emission_dim=10, latent_dim=2, rng=jr.PRNGKey(0), emissions="gaussian"):
    key1, key2 = jr.split(rng, 2)
    
    initial_distribution = tfp.distributions.MultivariateNormalTriL(
        np.zeros(latent_dim), np.eye(latent_dim))

    dynamics_distribution = GaussianLinearRegression(
        random_rotation(key1, latent_dim, theta=np.pi/20),
        np.zeros(latent_dim), 
        0.1**2 * np.eye(latent_dim))

    if emissions == "gaussian":
        emissions_distribution = GaussianLinearRegression(
            jr.normal(key2, shape=(emission_dim, latent_dim)), 
            np.zeros(emission_dim), 
            1.0**2 * np.eye(emission_dim))

        # Initialize our Gaussian LDS model
        lds = GaussianLDS(initial_distribution, 
                          dynamics_distribution,
                          emissions_distribution)

    elif emissions == "poisson":
        emissions_distribution = PoissonGLM(
            jr.normal(key2, shape=(emission_dim, latent_dim)), 
            np.zeros(emission_dim))

        lds = LDS(initial_distribution, 
                  dynamics_distribution, 
                  emissions_distribution)

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


#### LDS EM TESTS
@pytest.mark.parametrize("num_trials", range(1, 202, 50))
def test_lds_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (lds_fit_setup(num_trials=num_trials), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))
    
@pytest.mark.parametrize("num_timesteps", range(10, 20011, 10000))
def test_lds_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (lds_fit_setup(num_timesteps=num_timesteps), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))

@pytest.mark.parametrize("latent_dim", range(2, 13, 5))
def test_lds_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (lds_fit_setup(latent_dim=latent_dim), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))

@pytest.mark.parametrize("emissions_dim", range(2, 13, 5))
def test_lds_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (lds_fit_setup(emissions_dim=emissions_dim), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))


#### PLDS EM TESTS
@pytest.mark.parametrize("num_trials", range(1, 202, 50))
def test_lds_laplace_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (lds_fit_setup(num_trials=num_trials, emissions="poisson"), {})
    lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))
    
@pytest.mark.parametrize("num_timesteps", range(10, 1011, 250))
def test_lds_laplace_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (lds_fit_setup(num_timesteps=num_timesteps, emissions="poisson"), {})
    lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))

@pytest.mark.parametrize("latent_dim", range(2, 8, 5))
def test_lds_laplace_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (lds_fit_setup(latent_dim=latent_dim, emissions="poisson"), {})
    lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))

@pytest.mark.parametrize("emissions_dim", range(2, 8, 5))
def test_lds_laplace_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (lds_fit_setup(emissions_dim=emissions_dim, emissions="poisson"), {})
    lp = benchmark.pedantic(lds_fit_laplace_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))