"""
Base state-space model class.

In SSM-JAX, a state-space model is represented as a class with methods defining
the ``initial_distribution``, ``dynamics_distribution``, and ``emissions_distribution``
respectively.
"""

import jax.random as jr
from jax import lax, vmap
from jax.tree_util import tree_map

import tensorflow_probability.substrates.jax as tfp

from ssm.utils import format_dataset


class SSM(object):
    """
    A generic state-space model base class.
    """
    def initial_distribution(self) -> tfp.distributions.Distribution:
        """
        The distribution over the initial state of the SSM.

        .. math::
            p(x_1)

        Returns:
            initial_distribution (tfp.distributions.Distribution):
                A distribution over initial states in the SSM.
        """
        raise NotImplementedError

    def dynamics_distribution(self, state: float) -> tfp.distributions.Distribution:
        """
        The dynamics (or state-transition) distribution conditioned on the current state.

        .. math::
            p(x_{t+1} | x_t)

        Args:
            state: The current state on which to condition the dynamics.

        Returns:
            dynamics_distribution (tfp.distributions.Distribution):
                The distribution over states conditioned on the current state.
        """
        raise NotImplementedError

    def emissions_distribution(self, state: float) -> tfp.distributions.Distribution:
        """
        The emissions (or observation) distribution conditioned on the current state.

        .. math::
            p(y_t | x_t)

        Args:
            state: The current state on which to condition the emissions.

        Returns:
            emissions_distribution (tfp.distributions.Distribution):
                The emissions distribution conditioned on the provided state.
        """
        raise NotImplementedError

    def log_probability(self, states, data):
        r"""
        Computes the log joint probability of a set of states and data (observations).

        .. math::
            \log p(x, y) = \log p(x_1) + \sum_{t=1}^{T-1} \log p(x_{t+1} | x_t) + \sum_{t=1}^{T} \log p(y_t | x_t)

        Args:
            states: latent states :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{latent_dim})` 
            data: observed data :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{emissions_dim})` 

        Returns:
            lp: log joint probability :math:`\log p(x, y)`
                of shape :math:`(\text{batch]},)`
        """
        
        def _log_probability_single(_states, _data):
            lp = 0
            # Get the first timestep probability
            initial_state, initial_data = _states[0], _data[0]
            lp += self.initial_distribution().log_prob(initial_state)
            lp += self.emissions_distribution(initial_state).log_prob(initial_data)

            def _step(carry, args):
                prev_state, lp = carry
                state, emission = args
                lp += self.dynamics_distribution(prev_state).log_prob(state)
                lp += self.emissions_distribution(state).log_prob(emission)
                return (state, lp), None

            (_, lp), _ = lax.scan(_step, (initial_state, lp), (_states[1:], _data[1:]))
            return lp
        
        if data.ndim > 2:
            lp = vmap(_log_probability_single)(states, data)
        else:
            lp = _log_probability_single(states, data)
            
        return lp

    def sample(self, key: jr.PRNGKey, num_steps: int, covariates=None, initial_state=None, num_samples: int=1):
        r"""
        Sample from the joint distribution defined by the state space model.

        .. math::
            x, y \sim p(x, y)

        Args:
            key (jr.PRNGKey): A JAX pseudorandom number generator key.
            num_steps (int): Number of steps for which to sample.
            covariates: Optional covariates that may be needed for sampling.
                Default is None.
            initial_state: Optional state on which to condition the sampled trajectory.
                Default is None which samples the intial state from the initial distribution.
            num_samples (int): Number of indepedent samples (defines a batch dimension).

        Returns:
            states: an array of latent states across time :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{latent_dim})` 
            emissions: an array of observations across time :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{emissions_dim})` 
        """

        def _sample(key, covariates=None, initial_state=None):
            if initial_state is None:
                key1, key = jr.split(key, 2)
                initial_state = self.initial_distribution().sample(seed=key1)

            def _step(state, key):
                key1, key2 = jr.split(key, 2)
                emission = self.emissions_distribution(state).sample(seed=key1)
                next_state = self.dynamics_distribution(state).sample(seed=key2)
                return next_state, (state, emission)

            keys = jr.split(key, num_steps)
            _, (states, emissions) = lax.scan(_step, initial_state, keys)
            return states, emissions

        if num_samples > 1:
            batch_keys = jr.split(key, num_samples)
            states, emissions = vmap(_sample)(batch_keys, covariates, initial_state)
        else:
            states, emissions = _sample(key)

        return states, emissions
