import jax.experimental.optimizers as optimizers
import jax.numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit, value_and_grad, vmap
from matplotlib.gridspec import GridSpec
from ssm.distributions.glm import GaussianGLM, PoissonGLM
from ssm.distributions.linreg import GaussianLinearRegression
from ssm.models.lds import LDS, GaussianLDS
from ssm.plots import plot_dynamics_2d
from ssm.utils import random_rotation
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange
import warnings
from tqdm.auto import trange
from time import time

def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        t_elapsed = t2-t1
        print(f'Function {func.__name__!r} executed in {t_elapsed:.4f}s')
        return result, t_elapsed
    return wrap_func

def create_random_lds(emissions_dim=10, latent_dim=2, rng=jr.PRNGKey(0)):
    key1, key2 = jr.split(rng, 2)
    
    initial_distribution = tfp.distributions.MultivariateNormalTriL(
        np.zeros(latent_dim), np.eye(latent_dim))

    dynamics_distribution = GaussianLinearRegression(
        random_rotation(key1, latent_dim, theta=np.pi/20),
        np.zeros(latent_dim), 
        0.1**2 * np.eye(latent_dim))

    emissions_distribution = GaussianLinearRegression(
        jr.normal(key2, shape=(emissions_dim, latent_dim)), 
        np.zeros(emissions_dim), 
        1.0**2 * np.eye(emissions_dim))

    # Initialize our Gaussian LDS model
    lds = GaussianLDS(initial_distribution, 
                      dynamics_distribution,
                      emissions_distribution)
    
    return lds

def sample_lds(lds, num_trials=5, time_bins=200, rng=jr.PRNGKey(0)):
    all_states, all_data = lds.sample(key=rng, num_steps=time_bins, num_samples=num_trials)
    return all_states, all_data

def prep_lds_em_test(emission_dim=10, latent_dim=2, num_trials=5, time_bins=200, rng=jr.PRNGKey(0)):
    true_rng, sample_rng, test_rng = jr.split(rng, 3)

    # create and sample from true_lds
    true_lds = create_random_lds(emission_dim, latent_dim, rng=true_rng)
    states, data = sample_lds(true_lds, num_trials, time_bins, rng=sample_rng)
    
    # create test_lds and fit
    test_lds = create_random_lds(emission_dim, latent_dim, rng=test_rng)
    return true_lds, test_lds, states, data

def time_lds_em(emission_dim=10, latent_dim=2, num_trials=5, time_bins=200, num_iters=100, rng=jr.PRNGKey(0)):
    rng_prep, rng_fit = jr.split(rng, 2)
    true_lds, test_lds, states, data = prep_lds_em_test(emission_dim, latent_dim, num_trials, time_bins, rng_prep)
    (elbos, fitted_lds, posteriors), fit_time = timer(test_lds.fit)(data, method="em", num_iters=num_iters)
    return (elbos, fitted_lds, posteriors), fit_time