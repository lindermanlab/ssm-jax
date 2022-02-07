"""
Encoder templates for SMC (+FIVO).
"""
import jax
import jax.numpy as np
import flax.linen as nn
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd


class IndependentBiRnnEncoder:
    """
    NOTE - this bi-directional RNN encoder does not include a combinator head.  This is because
    we want to be able to use the forward message and backwards message separately.  This therefore
    basically just wraps up a call to a forward and backwards RNN.

    NOTE - the same RNN object is used in both directions, just with different parameters.
    """

    def __init__(self, _env, _key, _rnn, _single_dataset):

        self.latent_dim = _env.config.rnn_state_dim if _env.config.rnn_state_dim is not None else _env.config.latent_dim
        self._forward_rnn = _rnn
        self._backward_rnn = _rnn
        self.dummy_exposed_state = (self.initialize_carry(_key)[0][1], self.initialize_carry(_key)[1][1])

    def init(self, key, dummy_input):
        """
        Initialize the parameters of the RNNs.

        Args:
            key:
            dummy_input:

        Returns:

        """
        subkey1, subkey2 = jr.split(key)
        forward_params = self._forward_rnn.init(subkey1, dummy_input[0][0], dummy_input[1])
        backward_params = self._backward_rnn.init(subkey2, dummy_input[0][1], dummy_input[1])
        return (forward_params, backward_params)

    def initialize_carry(self, key, batch_dims=(), size=1, init_fn=nn.zeros):
        """

        Args:
            key:

        Returns:

        """
        f_carry = self._forward_rnn.initialize_carry(key, batch_dims=(), size=self.latent_dim)
        b_carry = self._backward_rnn.initialize_carry(key, batch_dims=(), size=self.latent_dim)
        return (f_carry, b_carry)

    def encode(self, _encoder_params, key, _single_dataset):
        """

        Args:
            _encoder_params:
            key:
            _single_dataset:

        Returns:

        """

        forward_params, backward_params = _encoder_params
        key, subkey = jr.split(key)
        init_forward_carry, init_backward_carry = self.initialize_carry(subkey)

        def forward_encoder_scan_fn(_old_carry, _input_at_t):
            _new_carry, _new_hidden_state = self._forward_rnn.apply(forward_params, _old_carry, _input_at_t)
            return _new_carry, _new_hidden_state

        def backward_encoder_scan_fn(_old_carry, _input_at_t):
            _new_carry, _new_hidden_state = self._backward_rnn.apply(backward_params, _old_carry, _input_at_t)
            return _new_carry, _new_hidden_state

        _, forward_encoded_data = jax.lax.scan(forward_encoder_scan_fn,
                                               init_forward_carry,
                                               _single_dataset)

        _, backward_encoded_data = jax.lax.scan(backward_encoder_scan_fn,
                                                init_backward_carry,
                                                _single_dataset,
                                                reverse=True)

        return (forward_encoded_data, backward_encoded_data)

    def __call__(self, *args, **kwargs):
        p = 0

    # def bind(self, _param_vals):
    #     """
    #
    #     Args:
    #         _param_vals:
    #
    #     Returns:
    #
    #     """
    #     self._forward_rnn.bind(_param_vals[0])
    #     self._backward_rnn.bind(_param_vals[1])


def rebuild_encoder(encoder, env):
    """
    """

    def _encode_data(_param_vals):

        # If there is no proposal, then there is no structure to define.
        if (encoder is None) or (env.config.encoder_structure == 'NONE'):
            return lambda *_args: None
        else:
            return lambda _key, _single_dataset: encoder.encode(_param_vals, _key, _single_dataset)

    return _encode_data
