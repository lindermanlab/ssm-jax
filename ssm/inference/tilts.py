"""
Tilt templates for SMC (+FIVO).
"""
import jax
import jax.numpy as np
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from copy import deepcopy as dc

# Import some ssm stuff.
from ssm.inference.conditional_generators import IndependentGaussianGenerator
import ssm.nn_util as nn_util


class IndependentGaussianTilt:
    """


    """

    def __init__(self, n_tilts, tilt_input,
                 trunk_fn=None, head_mean_fn=None, head_log_var_fn=None):

        # Work out the number of tilts.
        assert (n_tilts == 1) or (n_tilts == len(tilt_input[0]) - 1), \
            'Can only use a single tilt or as many tilt as there are transitions.'
        self.n_tilts = n_tilts

        # Re-build the full input that will be used to condition the tilt.
        self._dummy_processed_input = self._tilt_input_generator(*tilt_input)

        # Re-build the output that we will score under the tilt.
        dummy_output = self._tilt_output_generator(*tilt_input)
        output_dim = dummy_output.shape[0]

        # Build out the function approximator.
        self.tilt = IndependentGaussianGenerator.from_params(self._dummy_processed_input,
                                                             dummy_output,
                                                             trunk_fn=trunk_fn,
                                                             head_mean_fn=head_mean_fn,
                                                             head_log_var_fn=head_log_var_fn, )

    def init(self, key):
        """
        Initialize the parameters of the tilt distribution.

        Args:
            - key (jax.PRNGKey):    Seed to seed initialization.

        Returns:
            - parameters:           FrozenDict of the parameters of the initialized tilt.

        """
        return jax.vmap(self.tilt.init, in_axes=(0, None))\
            (jr.split(key, self.n_tilts), self._dummy_processed_input)

    def apply(self, params, inputs):
        """
        NOTE - this can be overwritten elsewhere for more specialist application requirements.

        Args:
            params (FrozenDict):    FrozenDict of the parameters of the proposal.

            inputs (tuple):         Tuple of standard inputs to the proposal in SMC:
                                    (dataset, model, particles, time, p_dist)

            data:

        Returns:
            (Float): Tilt log value.

        """

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = params[0]
        else:
            t = inputs[3]
            t_params = jax.tree_map(lambda args: args[t], params)

        # Generate a tilt distribution.
        tilt_inputs = self._tilt_input_generator(*inputs)
        r_dist = self.tilt.apply(t_params, tilt_inputs)

        # Now score under that distribution.
        tilt_outputs = self._tilt_output_generator(*inputs)
        log_r_val = r_dist.log_prob(tilt_outputs)

        return log_r_val

    def _tilt_input_generator(self, *_inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t) into a vector object that
        can be input into the tilt.

        NOTE - this is an abstract method and must be implemented by wherever this is being used.

        Args:
            *_inputs (tuple):       Tuple of standard inputs to the tilt in SMC:
                                    (dataset, model, particles, time)

        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into tilt.

        """
        raise NotImplementedError()

    def _tilt_output_generator(self, *_inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t) into a vector object that
        can be scored under into the tilt.

        NOTE - this is an abstract method and must be implemented by wherever this is being used.

        Args:
            *_inputs (tuple):       Tuple of standard inputs to the tilt in SMC:
                                    (dataset, model, particles, time)

        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into tilt.

        """
        raise NotImplementedError()


def rebuild_tilt(tilt, tilt_structure):
    """
    """

    def _rebuild_tilt(_param_vals):
        # If there is no tilt, then there is no structure to define.
        if tilt is None:
            return lambda *_: 0.0

        # We fork depending on the tilt type.
        # tilt takes arguments of (dataset, model, particles, time, p_dist, q_state, ...).
        if tilt_structure == 'DIRECT':

            def _tilt(*_input):
                #
                # # TODO - this has gotten a bit messy.
                # t = _input[3]
                #
                # p_all = _param_vals[t]  # jax.lax.dynamic_index_in_dim(np.asarray(tuple((0, 1, 2))), t, 0)
                #
                # p = jax.tree_map(lambda args: jax.lax.dynamic_index_in_dim(args, 0, 0, keepdims=False), p_all)
                #
                # r_log_val = tilt[t].apply(p, _input)

                r_log_val = tilt.apply(_param_vals, _input)

                return r_log_val
        else:
            raise NotImplementedError()

        return _tilt

    return _rebuild_tilt
