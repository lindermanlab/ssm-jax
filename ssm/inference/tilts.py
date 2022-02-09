"""
Tilt templates for SMC (+FIVO).
"""
import jax
import jax.numpy as np
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

# Import some ssm stuff.
from ssm.inference.conditional_generators import IndependentGaussianGenerator, IndependentBernoulliGenerator
import ssm.nn_util as nn_util


class IndependentGaussianTilt:
    """


    """

    def __init__(self, n_tilts, tilt_input,
                 trunk_fn=None, head_mean_fn=None, head_log_var_fn=None, distribution_class='GAUSSIAN'):

        # Work out the number of tilts.
        assert (n_tilts == 1) or (n_tilts == len(tilt_input[0]) - 1), \
            'Can only use a single tilt or as many tilt as there are transitions.'
        self.n_tilts = n_tilts

        # This property can be overwritten by child classes.
        self.tilt_window_length = tilt_input[4]

        # Re-build the full input that will be used to condition the tilt.
        self._dummy_processed_input = self._tilt_input_generator(*tilt_input)

        # Re-build the output that we will score under the tilt.
        dummy_output = self._tilt_output_generator(*tilt_input)

        # Build out the function approximator.
        if distribution_class == 'GAUSSIAN':
            self.tilt = IndependentGaussianGenerator.from_params(self._dummy_processed_input,
                                                                 dummy_output,
                                                                 trunk_fn=trunk_fn,
                                                                 head_mean_fn=head_mean_fn,
                                                                 head_log_var_fn=head_log_var_fn, )
        elif distribution_class == 'BERNOULLI':
            assert head_mean_fn is None, "Cannot specify head functions here.  Use trunk instead."
            assert head_log_var_fn is None, "Cannot specify head functions here.  Use trunk instead."
            self.tilt = IndependentBernoulliGenerator.from_params(self._dummy_processed_input,
                                                                  dummy_output,
                                                                  trunk_fn=trunk_fn, )
        else:
            raise NotImplementedError()

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

    def apply(self, params, dataset, model, particles, t, *inputs):
        """
        NOTE - this can be overwritten elsewhere for more specialist application requirements.

        Args:
            params (FrozenDict):    FrozenDict of the parameters of the tilt.

            dataset:

            model:

            particles:

            t:

            *inputs

        Returns:
            (Float): Tilt log value.

        """

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = jax.tree_map(lambda args: args[0], params)  # params[0]
        else:
            t_params = jax.tree_map(lambda args: args[t], params)

        # Generate a tilt distribution.
        tilt_inputs = self._tilt_input_generator(dataset, model, particles, t, *inputs)
        r_dist = self.tilt.apply(t_params, tilt_inputs)

        # Now score under that distribution.
        tilt_outputs = self._tilt_output_generator(dataset, model, particles, t, self.tilt_window_length, *inputs)
        log_r_val = r_dist.log_prob(tilt_outputs)


        # # TODO - removed tilt.
        # log_r_val = log_r_val * 0.0
        # # TODO - removed tilt.


        return log_r_val

    # Define a method for generating thei nputs to the tilt.
    def _tilt_input_generator(self, dataset, model, particles, t, *inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t) into a vector object that
        can be input into the tilt.

        NOTE - because of the conditional independncies introduced by the HMM, there is no dependence
        on the previous states.

        NOTE - this function may be overriden in child classes to provide more sophisticated inputs.

        Args:
            dataset:

            model:

            particles:

            t:

            *inputs:

        Returns:
            (ndarray):              Processed and vectorized version of `*inputs` ready to go into tilt.

        """

        # Just the particles are passed in.
        tilt_inputs = (particles, )

        model_latent_shape = (model.latent_dim, )

        is_batched = (model_latent_shape != particles.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(tilt_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(0, ))
            return vmapped(*tilt_inputs)

    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, tilt_window_length, *inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t) into a vector object that
        can be scored under into the tilt.

        NOTE - this is an abstract method and must be implemented by wherever this is being used.

        Args:
            *inputs (tuple):       Tuple of standard inputs to the tilt in SMC:
                                    (dataset, model, particles, time)

        Returns:
            (ndarray):              Processed and vectorized version of `*inputs` ready to go into tilt.

        """
        # We will pass in whole data into the tilt and then filter out as required.
        tilt_outputs = (dataset, )
        return nn_util.vectorize_pytree(tilt_outputs)


class IGPerStepTilt(IndependentGaussianTilt):
    """

    """

    def apply(self, params, dataset, model, particles, t, *inputs):
        """

        """

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = jax.tree_map(lambda args: args[0], params)
        else:
            t_params = jax.tree_map(lambda args: args[t], params)

        # Generate a tilt distribution.
        tilt_inputs = self._tilt_input_generator(dataset, model, particles, t, *inputs)
        r_dist = self.tilt.apply(t_params, tilt_inputs)

        # Now score under that distribution.
        tilt_outputs = self._tilt_output_generator(dataset, model, particles, t, self.tilt_window_length, *inputs)

        # There may be NaNs here, so we need to pull this apart.
        means = r_dist.mean().T
        sds = r_dist.variance().T

        # Sweep over the vector and return zeros where appropriate.
        def _eval(_idx, _mu, _sd, _out):
            _dist = tfd.MultivariateNormalDiag(loc=np.expand_dims(_mu, -1), scale_diag=np.sqrt(np.expand_dims(_sd, -1)))

            # Define the difference scoring criteria.
            def _score_all_future():
                return _idx > t  # Pretty sure this should be a <= (since we are scoring _future_ observations).

            return jax.lax.cond(_score_all_future(),
                                lambda *args: _dist.log_prob(np.asarray([_out])),
                                lambda *args: np.zeros_like(_dist.log_prob(np.asarray([_out]))),
                                None)

        log_r_val = jax.vmap(_eval)(np.arange(means.shape[0]), means, sds, tilt_outputs).sum(axis=0)
        return log_r_val


class IGWindowTilt(IndependentGaussianTilt):

    # We need to define the method for generating the inputs.
    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, _tilt_window_length, *_):
        """

        """
        # # TODO - this behaviour should be discouraged to prevent erroroneous settings, but let it slide for the time being.
        # assert (inputs == ()) or (inputs == (None, )), "Cannot supply inputs with raw window."

        masked_idx = np.arange(_tilt_window_length)
        to_insert = (t + 1 + masked_idx < len(dataset))  # We will insert where the window is inside the dataset.

        # Zero out the elements outside of the valid range.
        clipped_dataset = jax.lax.dynamic_slice(dataset,
                                                (t+1, *tuple(0 * _d for _d in dataset.shape[1:])),
                                                (_tilt_window_length, *dataset.shape[1:]))
        masked_dataset = clipped_dataset * np.expand_dims(to_insert.astype(np.int32), 1)

        # We will pass in whole data into the tilt and then filter out as required.
        tilt_outputs = (masked_dataset, )
        return nn_util.vectorize_pytree(tilt_outputs)

    def apply(self, params, dataset, model, particles, t, *inputs):
        """

        Args:
            params:
            dataset:
            model:
            particles:
            t:
            *inputs:

        Returns:

        """

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = jax.tree_map(lambda args: args[0], params)
        else:
            t_params = jax.tree_map(lambda args: args[t], params)

        # Generate a tilt distribution.
        tilt_inputs = self._tilt_input_generator(dataset, model, particles, t, *inputs)
        r_dist = self.tilt.apply(t_params, tilt_inputs)

        # Now score under that distribution.
        tilt_outputs = self._tilt_output_generator(dataset, model, particles, t, self.tilt_window_length, *inputs)

        # TODO - need to work out if there is a continouous / discrete tilt in here.

        # We need to re-capitulate the distribution for each of the timesteps (its independent) and accumulate
        # just those within the time window.
        means = r_dist.mean()
        variances = r_dist.variance()
        dataset_length = len(dataset)
        mask = (np.repeat(np.expand_dims(np.arange(self.tilt_window_length) + t + 1, 1), dataset.shape[-1], axis=1) < dataset_length).flatten()

        def _eval(_mu, _var, _mask, _obs):
            return tfd.Normal(_mu, np.sqrt(_var)).log_prob(_obs) * _mask

        is_batched = (means.shape != mask.shape)
        if is_batched:
            r_vals = jax.vmap(_eval, in_axes=(1, 1, 0, 0))(means, variances, mask, tilt_outputs)
        else:
            r_vals = _eval(means, variances, mask, tilt_outputs)
        log_r_val = np.sum(r_vals, axis=0)


        # # TODO - removed tilt.
        # log_r_val = log_r_val * 0.0
        # # TODO - removed tilt.


        return log_r_val


def rebuild_tilt(tilt, env):
    """
    """

    def _rebuild_tilt(_param_vals, _dataset, _model, _encoded_data=None):
        # If there is no tilt, then there is no structure to define.
        if tilt is None:
            return lambda *_: 0.0

        # We fork depending on the tilt type.
        # tilt takes arguments of (dataset, model, particles, time, p_dist, q_state, ...).
        if env.config.tilt_structure == 'DIRECT':

            def _tilt(_particles, _t, __dataset=None, __encoded_data=None):

                if __dataset is None:
                    __dataset = _dataset

                if __encoded_data is None:
                    __encoded_data = _encoded_data

                r_log_val = tilt.apply(_param_vals, __dataset, _model, _particles, _t, __encoded_data)
                return r_log_val
        else:
            raise NotImplementedError()

        return _tilt

    return _rebuild_tilt
