"""
Base state-space model class.

In SSM-JAX, a state-space model is represented as a class with methods defining
the ``initial_distribution``, ``dynamics_distribution``, and ``emissions_distribution``
respectively.

The base ``SSM`` object provides template functionality for a state space model.
"""
import jax.numpy as np
import jax.random as jr
from jax import lax, vmap

import tensorflow_probability.substrates.jax as tfp

from ssm.utils import tree_get, auto_batch


class SSM(object):
    """
    A generic state-space model base class.
    """
    def initial_distribution(self,
                             covariates=None,
                             metadata=None) -> tfp.distributions.Distribution:
        """
        The distribution over the initial state of the SSM.

        .. math::
            p(x_1)

        Returns:
            initial_distribution (tfp.distributions.Distribution):
                A distribution over initial states in the SSM.
        """
        raise NotImplementedError

    def dynamics_distribution(self,
                              state: float,
                              covariates=None,
                              metadata=None) -> tfp.distributions.Distribution:
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

    def emissions_distribution(self,
                               state: float,
                               covariates=None,
                               metadata=None) -> tfp.distributions.Distribution:
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

    @property
    def emissions_shape(self):
        """
        Returns the shape of a single emission, :math:`y_t`.

        Returns:
            A tuple or tree of tuples giving the emission shape(s).
        """
        raise NotImplementedError

    @auto_batch(batched_args=("states", "data", "covariates", "metadata"))
    def log_probability(self,
                        states,
                        data,
                        covariates=None,
                        metadata=None):
        r"""
        Computes the log joint probability of a set of states and data (observations).

        .. math::
            \log p(x, y) = \log p(x_1) + \sum_{t=1}^{T-1} \log p(x_{t+1} | x_t) + \sum_{t=1}^{T} \log p(y_t | x_t)

        Args:
            states: latent states :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{latent\_dim})`
            data: observed data :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`

        Returns:
            lp: log joint probability :math:`\log p(x, y)`
                of shape :math:`(\text{batch]},)`
        """

        lp = 0

        # Get the first timestep probability
        initial_state = tree_get(states, 0)
        initial_data = tree_get(data, 0)
        initial_covariates = tree_get(covariates, 0)

        lp += self.initial_distribution(
            covariates=initial_covariates, metadata=metadata).log_prob(initial_state)
        lp += self.emissions_distribution(
            initial_state, covariates=initial_covariates, metadata=metadata).log_prob(initial_data)

        def _step(carry, args):
            prev_state, lp = carry
            state, emission, covariates = args
            lp += self.dynamics_distribution(
                prev_state, covariates=covariates, metadata=metadata).log_prob(state)
            lp += self.emissions_distribution(
                state, covariates=covariates, metadata=metadata).log_prob(emission)
            return (state, lp), None

        (_, lp), _ = lax.scan(_step, (initial_state, lp),
                                (tree_get(states, slice(1, None)),
                                tree_get(data, slice(1, None)),
                                tree_get(covariates, slice(1, None))))
        return lp

    @auto_batch(batched_args=("key", "data", "posterior", "covariates", "metadata"))
    def elbo(self,
             key,
             data,
             posterior,
             covariates=None,
             metadata=None,
             num_samples=1):
        """
        Compute an _evidence lower bound_ (ELBO) using the joint probability and an
        approximate posterior :math:`q(x) \\approx p(x | y)`:

        .. math:
            log p(y) \geq \mathbb{E}_q \left[\log p(y, x) - \log q(x) \\right]

        While in some cases the expectation can be computed in closed form, in
        general we will approximate it with ordinary Monte Carlo.
        """
        def _elbo_single(_key):
            sample = posterior.sample(seed=_key)
            return self.log_probability(sample, data, covariates, metadata) - posterior.log_prob(sample)

        elbos = vmap(_elbo_single)(jr.split(key, num_samples))
        return np.mean(elbos)

    def sample(self,
               key: jr.PRNGKey,
               num_steps: int,
               initial_state=None,
               covariates=None,
               metadata=None,
               num_samples: int=1):
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
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{latent\_dim})`
            emissions: an array of observations across time :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
        """

        def _sample(key, covariates=None, initial_state=None):

            if initial_state is None:
                key1, key = jr.split(key, 2)
                initial_covariates = tree_get(covariates, 0)
                initial_state = self.initial_distribution(covariates=initial_covariates,
                                                          metadata=metadata).sample(seed=key1)

            def _step(state, key_and_covariates):
                key, covariates = key_and_covariates
                key1, key2 = jr.split(key, 2)
                emission = self.emissions_distribution(state,
                                                       covariates=covariates,
                                                       metadata=metadata).sample(seed=key1)
                next_state = self.dynamics_distribution(state,
                                                        covariates=covariates,
                                                        metadata=metadata).sample(seed=key2)
                return next_state, (state, emission)

            keys = jr.split(key, num_steps)
            _, (states, emissions) = lax.scan(_step, initial_state, (keys, covariates))
            return states, emissions

        if num_samples > 1:
            batch_keys = jr.split(key, num_samples)
            states, emissions = vmap(_sample)(batch_keys, covariates, initial_state)
        else:
            states, emissions = _sample(key, covariates, initial_state)

        return states, emissions
