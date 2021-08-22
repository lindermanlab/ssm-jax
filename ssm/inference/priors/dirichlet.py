import jax.numpy as np
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp


def _dirichlet_from_stats(stats, counts):
    concentration = stats[0]
    return tfp.distributions.Dirichlet(concentration)


def _dirichlet_pseudo_obs_and_counts(dirichlet):
    return (dirichlet.concentration,), 0


def _categorical_from_params(params):
    return tfp.distributions.Categorical(probs=params)


def _categorical_suff_stats(data):
    num_classes = int(data.max()) + 1
    return (data[..., None] == np.arange(num_classes),)
