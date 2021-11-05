"""
Base class for Autoregressive HMM.
"""
import jax.numpy as np
import jax.random as jr
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class

from ssm.hmm.base import HMM

@register_pytree_node_class
class AutoregressiveHMM(HMM):
    r"""Base class for HMM with autoregressive dependencies.
    
    Inherits from HMM base class.
    """
    
    @property
    def emission_dim(self):
        return self._emissions._distribution.data_dimension

    @property
    def num_lags(self):
        dist = self._emissions._distribution
        return dist.covariate_dimension // dist.data_dimension

    def log_probability(self, states, data, prev_emissions=None):
        r"""
        Computes the log joint probability of a set of states and data (observations).

        .. math::
            \log p(x, y) = \log p(x_1) + \sum_{t=1}^{T-1} \log p(x_{t+1} | x_t) + \sum_{t=1}^{T} \log p(y_t | x_t)

        Args:
            states: latent states :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{latent_dim})` 
            data: observed data :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{emissions_dim})` 
            prev_emissions: previous emissions to condition on
                of shape :math:`(\text{[batch]} , \text{num_lags} , \text{emissions_dim})`.
                Default is None which will condition on zeros.

        Returns:
            lp: log joint probability :math:`\log p(x, y)`
                of shape :math:`(\text{batch]},)`
        """
        
        def _log_probability_single(_states, _data, _prev_emissions):
            if _prev_emissions is None:
                _prev_emissions = np.zeros((self.num_lags, self.emission_dim))

            lp = 0
            lp += self.initial_distribution().log_prob(_states[0])
            lp += self.emissions_distribution(_states[0]).log_prob(_data[0], covariates=_prev_emissions.ravel())

            def _step(carry, args):
                prev_state, _prev_emissions, lp = carry
                state, emission = args
                lp += self.dynamics_distribution(prev_state).log_prob(state)
                lp += self.emissions_distribution(state).log_prob(emission, covariates=_prev_emissions.ravel())
                new_prev_emissions = np.row_stack([_prev_emissions[1:], emission])
                return (state, new_prev_emissions, lp), None

            initial_carry = (_states[0], np.row_stack([_prev_emissions[1:], _data[0]]), lp)
            (_, _, lp), _ = lax.scan(_step, initial_carry, (_states[1:], _data[1:]))
            return lp
        
        if data.ndim > 2:
            lp = vmap(_log_probability_single)(states, data, prev_emissions)
        else:
            lp = _log_probability_single(states, data, prev_emissions)
            
        return lp


    def sample(self, key, num_steps: int, initial_state=None, num_samples: int=1, prev_emissions=None):
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
            prev_emissions: previous emissions to condition on
                of shape :math:`(\text{[batch]} , \text{num_lags} , \text{emissions_dim})`.
                Default is None which will condition on zeros.

        Returns:
            states: an array of latent states across time :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{latent_dim})` 
            emissions: an array of observations across time :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num_lags} , \text{emissions_dim})`.
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
