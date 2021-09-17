"""
Base state-space model class.
"""

import jax.random as jr
from jax import lax


class SSM(object):
    """ 
    A generic state-space model base class.

    In ``SSM-JAX``, a state-space model is represented as a class with methods defining
    the ``initial_distribution``, ``dynamics_distribution``, and ``emissions_distribution``
    respectively.
    """
    def initial_distribution(self):
        """
        The distribution over the initial state of the SSM.

        Returns:
            A distribution over initial states in the SSM. 
        """
        raise NotImplementedError

    def dynamics_distribution(self, state: float):
        """
        The dynamics (or state-transition) distribution conditioned on the current state.

        Args:
            state: The current state on which to condition the dynamics.

        Returns:
            A distribution over the subsequent states in the SSM.
        """
        raise NotImplementedError

    def emissions_distribution(self, state: float):
        """
        The emissions (or observation) distribution conditioned on the current state.

        Args:
            state: The current state on which to condition the emissions.

        Returns:
            An emissions distribution conditioned on the provided state.
        """
        raise NotImplementedError

    def log_probability(self, states, data):
        """
        Computes the log joint probability of a set of states and data.

        The log probability of a state-space model is effectively a joint
        distribution over the (latent) states and observations.

        Args:
            states: An array of latent states.
            data: An array of the observed data.

        Returns:
            The joint log probability of the provided states and data.
        """
        lp = 0
        lp += self.initial_distribution().log_prob(states[0])
        lp += self.emissions_distribution(states[0]).log_prob(data[0])

        def _step(carry, args):
            prev_state, lp = carry
            state, emission = args
            lp += self.dynamics_distribution(prev_state).log_prob(state)
            lp += self.emissions_distribution(state).log_prob(emission)
            return (state, lp), None

        (_, lp), _ = lax.scan(_step, (states[0], lp), (states[1:], data[1:]))
        return lp

    def sample(self, key, num_steps: int, covariates=None, initial_state=None):
        """
        Sample the state space model for specified number of steps.

        Args:
            key (PRNGKey): A JAX pseudorandom number generator key.
            num_steps (int): Number of steps for which to sample.
            covariates: Optional covariates that may be needed for sampling.
            initial_state: Optional state to explicitly condition on (default is None,
                which means the initial state is sampled from the initial state 
                distribution.

        """
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
