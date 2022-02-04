from jax.tree_util import register_pytree_node_class
import jax.random as jr
import jax.numpy as np
import flax.linen as nn

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from ssm.rnn.base import DeterministicRNN


@register_pytree_node_class
class GRU(DeterministicRNN):
    def __init__(
        self,
        num_input_dims,
        num_latent_dims,
        initial_state=None,
        rnn_params=None,
        seed=None,
    ):
        if rnn_params is None:
            assert seed is not None, "must provide a random seed"
            rng_1, rng_2, seed = jr.split(seed, 3)
            carry = nn.GRUCell().initialize_carry(
                rng_1, batch_dims=tuple(), size=num_latent_dims
            )
            dummy_data = np.zeros((num_input_dims,))
            rnn_params = nn.GRUCell().init(rng_2, carry, dummy_data)

        if initial_state is None:
            assert seed is not None, "must provide a random seed"
            rng_1, seed = jr.split(seed, 2)
            initial_state = jr.normal(rng_1, shape=(num_latent_dims,))
            
        self.num_input_dims = num_input_dims
        self.num_latent_dims = num_latent_dims
            
        super(GRU, self).__init__(rnn_params, initial_state)
    
    @property
    def emissions_shape(self):
        return (self.num_latent_dims,)

    def dynamics_distribution(self, state, covariates=None, metadata=None):
        _, new_state = nn.GRUCell().apply(self._rnn_params, state, covariates)
        return tfd.VectorDeterministic(new_state, atol=1e-5)

@register_pytree_node_class
class LSTM(DeterministicRNN):
    def __init__(
        self,
        num_input_dims,
        num_latent_dims,
        initial_state=None,
        rnn_params=None,
        seed=None,
    ):
        if rnn_params is None:
            assert seed is not None, "must provide a random seed"
            rng_1, rng_2, seed = jr.split(seed, 3)
            carry = nn.LSTMCell().initialize_carry(
                rng_1, batch_dims=tuple(), size=num_latent_dims
            )
            dummy_data = np.zeros((num_input_dims,))
            rnn_params = nn.LSTMCell().init(rng_2, carry, dummy_data)

        if initial_state is None:
            assert seed is not None, "must provide a random seed"
            rng_1, seed = jr.split(seed, 2)
            initial_state = jr.normal(rng_1, shape=(num_latent_dims,))
        super(LSTM, self).__init__(rnn_params, initial_state)
        
    @property
    def emissions_shape(self):
        return (self.num_latent_dims,)

    def dynamics_distribution(self, state, covariates=None, metadata=None):
        _, new_state = nn.LSTMCell().apply(self._rnn_params, state, covariates)
        return tfd.VectorDeterministic(new_state, atol=1e-5)