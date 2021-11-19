from functools import partial

import jax.numpy as np
import jax.random as jr
import jax.nn as nn
from jax.tree_util import register_pytree_node_class
from jax import vmap

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.hmm.initial import StandardInitialCondition as DiscreteInitialCondition
from ssm.hmm.transitions import StationaryTransitions
from ssm.lds.initial import StandardInitialCondition as ContinuousInitialCondition
from ssm.slds.base import SLDS
from ssm.slds.dynamics import StandardDynamics
from ssm.slds.emissions import GaussianEmissions
from ssm.utils import ensure_has_batch_dim, random_rotation


@register_pytree_node_class
class GaussianSLDS(SLDS):
    def __init__(self,
                 num_states: int,
                 latent_dim: int,
                 emission_dim: int,
                 initial_probs: np.ndarray=None,
                 initial_mean: np.ndarray=None,
                 initial_scale_tril: np.ndarray=None,
                 transition_matrix: np.ndarray=None,
                 dynamics_weights: np.ndarray=None,
                 dynamics_biases: np.ndarray=None,
                 dynamics_scale_trils: np.ndarray=None,
                 emission_weights: np.ndarray=None,
                 emission_biases: np.ndarray=None,
                 emission_scale_trils: np.ndarray=None,
                 seed: jr.PRNGKey=None):
        """SLDS with Gaussian emissions.

        The GaussianSLDS can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_states``, ``latent_dim``, and ``emission_dim``,
        and ``seed`` to create a GaussianSLDS with generic, randomly initialized parameters.

        Args:
            num_states (int): number of discrete latent states
            latent_dim (int): number of continuous latent dimensions
            emission_dim (int, optional): number of emissions dimensions.
            initial_probs (np.ndarray, optional): [description]. Defaults to None.
            initial_mean (np.ndarray, optional): [description]. Defaults to None.
            initial_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            dynamics_weights (np.ndarray, optional): [description]. Defaults to None.
            dynamics_biases (np.ndarray, optional): [description]. Defaults to None.
            dynamics_scale_trils (np.ndarray, optional): [description]. Defaults to None.
            emission_weights (np.ndarray, optional): [description]. Defaults to None.
            emission_bias (np.ndarray, optional): [description]. Defaults to None.
            emission_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            seed (jr.PRNGKey, optional): [description]. Defaults to None.
        """
        if initial_probs is None:
            initial_probs = np.ones(num_states) / num_states

        if initial_mean is None:
            initial_mean = np.zeros(latent_dim)

        if initial_scale_tril is None:
            initial_scale_tril = np.eye(latent_dim)

        if transition_matrix is None:
            transition_matrix = np.ones((num_states, num_states)) / num_states

        if dynamics_weights is None:
            seed, rng = jr.split(seed, 2)
            dynamics_weights = vmap(partial(random_rotation, n=latent_dim, theta=np.pi/20),
                                    jr.split(rng, num_states))

        if dynamics_biases is None:
            seed, rng = jr.split(seed, 2)
            dynamics_biases = jr.normal(rng, (num_states, latent_dim))

        if dynamics_scale_trils is None:
            dynamics_scale_trils = 0.1 * np.eye(latent_dim)

        if emission_weights is None:
            seed, rng = jr.split(seed, 2)
            emission_weights = jr.normal(rng, shape=(emission_dim, latent_dim))

        if emission_biases is None:
            emission_biases = np.zeros(emission_dim)

        if emission_scale_trils is None:
            emission_scale_trils = 1.0**2 * np.eye(emission_dim)

        # Initialize the components
        discrete_initial_condition = \
            DiscreteInitialCondition(num_states, initial_probs)

        continuous_initial_condition = \
            ContinuousInitialCondition(initial_mean=initial_mean,
                                       initial_scale_tril=initial_scale_tril)

        transitions = StationaryTransitions(num_states,
                                            transition_matrix=transition_matrix)

        dynamics = StandardDynamics(weights=dynamics_weights,
                                    biases=dynamics_biases,
                                    scale_trils=dynamics_scale_trils)

        emissions = GaussianEmissions(weights=emission_weights,
                                      bias=emission_biases,
                                      scale_tril=emission_scale_trils)

        super().__init__(num_states,
                         latent_dim,
                         discrete_initial_condition,
                         continuous_initial_condition,
                         transitions,
                         dynamics,
                         emissions)

    # Methods for inference
    # def infer_posterior(self, data):
    #     raise NotImplementedError

    # @format_dataset
    # def fit(self, dataset, method="em", rng=None, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG):
    #     r"""Fit the GaussianLDS to a dataset using the specified method.

    #     Note: because the observations are Gaussian, we can perform exact EM for a GaussianEM
    #     (i.e. the model is conjugate).

    #     Args:
    #         dataset (np.ndarray): observed data
    #             of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{emissions_dim})`
    #         method (str, optional): model fit method.
    #             Must be one of ["em", "laplace_em"]. Defaults to "em".
    #         rng (jr.PRNGKey, optional): Random seed.
    #             Defaults to None.
    #         num_iters (int, optional): number of fit iterations.
    #             Defaults to 100.
    #         tol (float, optional): tolerance in log probability to determine convergence.
    #             Defaults to 1e-4.
    #         verbosity (Verbosity, optional): print verbosity.
    #             Defaults to Verbosity.DEBUG.

    #     Raises:
    #         ValueError: if fit method is not reocgnized

    #     Returns:
    #         elbos (np.ndarray): elbos at each fit iteration
    #         model (LDS): the fitted model
    #         posteriors (LDSPosterior): the fitted posteriors
    #     """
    #     model = self
    #     kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

    #     if method == "em":
    #         elbos, lds, posteriors = em(model, dataset, **kwargs)
    #     elif method == "laplace_em":
    #         if rng is None:
    #             raise ValueError("Laplace EM requires a PRNGKey. Please provide an rng to fit.")
    #         elbos, lds, posteriors = laplace_em(rng, model, dataset, **kwargs)
    #     else:
    #         raise ValueError(f"Method {method} is not recognized/supported.")

    #     return elbos, lds, posteriors


