"""
HMM Model Classes
=================

Module defining model behavior for Hidden Markov Models (HMMs).
"""
from typing import Any
Array = Any

import jax.numpy as np
import jax.random as jr
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class

from ssm.hmm.base import HMM

@register_pytree_node_class
class AutoregressiveHMM(HMM):
    """
    TODO
    """
    @property
    def emission_dim(self):
        return self._emissions._emission_distribution.data_dimension

    @property
    def num_lags(self):
        dist = self._emissions._emission_distribution
        return dist.covariate_dimension // dist.data_dimension

    def log_probability(self, states, data, prev_emissions=None):
        """
        Computes the log joint probability of a set of states and data (observations).

        .. math::
            \log p(x, y) = \log p(x_1) + \sum_{t=1}^{T-1} \log p(x_{t+1} | x_t) + \sum_{t=1}^{T} \log p(y_t | x_t)

        Args:
            states: An array of latent states (:math:`x_{1:T}`).
            data: An array of the observed data (:math:`y_{1:T}`).

        Returns:
            lp:
                The joint log probability of the provided states and data.
        """
        if prev_emissions is None:
            prev_emissions = np.zeros((self.num_lags, self.emission_dim))

        lp = 0
        lp += self.initial_distribution().log_prob(states[0])
        lp += self.emissions_distribution(states[0]).log_prob(data[0], covariates=prev_emissions.ravel())

        def _step(carry, args):
            prev_state, prev_emissions, lp = carry
            state, emission = args
            lp += self.dynamics_distribution(prev_state).log_prob(state)
            lp += self.emissions_distribution(state).log_prob(emission, covariates=prev_emissions.ravel())
            new_prev_emissions = np.row_stack([prev_emissions[1:], emission])
            return (state, new_prev_emissions, lp), None

        initial_carry = (states[0], np.row_stack([prev_emissions[1:], data[0]]), lp)
        (_, _, lp), _ = lax.scan(_step, initial_carry, (states[1:], data[1:]))
        return lp


    def sample(self, key, num_steps: int, initial_state=None, num_samples=1, prev_emissions=None):
        """
        Sample from the joint distribution defined by the state space model.

        .. math::
            x, y \sim p(x, y)

        Args:
            key (PRNGKey): A JAX pseudorandom number generator key.
            num_steps (int): Number of steps for which to sample.
            initial_state: Optional state on which to condition the sampled trajectory.
                Default is None which samples the intial state from the initial distribution.
            prev_emissions: Optional initial emissions to start the autoregressive model.

        Returns:
            states: A ``(timesteps,)`` array of the state value across time (:math:`x`).
            emissions: A ``(timesteps, obs_dim)`` array of the observations across time (:math:`y`).

        """

        def _sample(key):
            if initial_state is None:
                key1, key = jr.split(key, 2)
                state = self.initial_distribution().sample(seed=key1)
            else:
                state = initial_state

            if prev_emissions is None:
                history = np.zeros((self.num_lags, self.emission_dim))
            else:
                history = prev_emissions

            def _step(carry, key):
                history, state = carry
                key1, key2 = jr.split(key, 2)
                emission = self.emissions_distribution(state).sample(seed=key1, covariates=history.ravel())
                next_state = self.dynamics_distribution(state).sample(seed=key2)
                next_history = np.row_stack([history[1:], emission])
                return (next_history, next_state), (state, emission)

            keys = jr.split(key, num_steps)
            _, (states, emissions) = lax.scan(_step, (history, state), keys)
            return states, emissions

        if num_samples > 1:
            batch_keys = jr.split(key, num_samples)
            states, emissions = vmap(_sample)(batch_keys)
        else:
            states, emissions = _sample(key)

        return states, emissions
