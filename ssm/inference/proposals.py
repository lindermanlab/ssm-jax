"""
Proposal templates for SMC (+FIVO).
"""
import jax
import jax.numpy as np
import flax.linen as nn
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

# Import some ssm stuff.
from ssm.inference.conditional_generators import IndependentGaussianGenerator
import ssm.nn_util as nn_util


class IndependentGaussianProposal:
    """
    Define a proposal FUNCTION that is an indpendent gaussian.  This Module is actually just a thin wrapper over a set
    of Linen modules.

    To define a different proposal FUNCTION here (as opposed to a proposal structure), change this class.

    This modules `.apply` method also wraps a call to the input generator, which takes the standard proposal
    parameterization (dataset, model, particles, t, p_dist, q_state, *inputs) and flattens the right form.

    """

    def __init__(self, n_proposals, stock_proposal_input, dummy_output,
                 trunk_fn=None, head_mean_fn=None, head_log_var_fn=None, proposal_window_length=None):
        """
        Args:
            n_proposals (int):
                Number of independent proposals to define.  Here must be equal to one (other definitions may use multiple
                proposals independently indexed by discrete state/time etc.

            stock_proposal_input (tuple):
                Tuple of the stock proposal input used in SMC.  Of type: (dataset, model, particles, time, p_dist).

            dummy_output (ndarray):
                Correctly shaped ndarray of dummy values used to define the output layer.

            trunk_fn (nn.Module):
                Linen module that is applied to the prepared inputs to create a common encoding.

            head_mean_fn (nn.Module):
                Linen module that is applied to the common encoding to create the Gaussian mean.

            head_log_var_fn (nn.Module):
                Linen module that is applied to the common encoding to create the log variance of the Gaussian.

            proposal_window_length (int):
                If using a windowed proposal, this specified the length of the window.  Note that to use a windowed proposal the
                input generator function must be updated, and so this is just declared here to define a consistent interface.

        Returns:
            None

        """
        # Work out the number of proposals.
        assert (n_proposals == 1) or (n_proposals == 2) or (n_proposals == len(stock_proposal_input[0])), \
            'Can only use a single proposal, two proposals (init and single window), or as many proposals as there are states.'
        self.n_proposals = n_proposals

        self.proposal_window_length = proposal_window_length

        # Re-build the full input that will be provided.
        self._dummy_processed_input = self._proposal_input_generator(*stock_proposal_input)[0]
        output_dim = nn_util.vectorize_pytree(dummy_output).shape[0]

        # Build out the function approximator.
        self.proposal = IndependentGaussianGenerator.from_params(self._dummy_processed_input,
                                                                 dummy_output,
                                                                 trunk_fn=trunk_fn,
                                                                 head_mean_fn=head_mean_fn,
                                                                 head_log_var_fn=head_log_var_fn, )

    def init(self, key):
        """
        Initialize the parameters of the proposal distribution.

        Args:
            key (jax.PRNGKey):    Seed to seed initialization.

        Returns:
            parameters:           FrozenDict of the parameters of the initialized proposal.

        """
        # return self.proposal.init(key, self._dummy_processed_input)
        return jax.vmap(self.proposal.init, in_axes=(0, None))\
            (jr.split(key, self.n_proposals), self._dummy_processed_input)

    def apply(self, params, dataset, model, particles, t, p_dist, q_state, *inputs):
        """

        Args:
            params (FrozenDict):    FrozenDict of the parameters of the proposal.
            dataset:
            model:
            particles:
            t:
            p_dist:
            q_state:

        Returns:
            (Tuple): (TFP distribution over latent state, updated q internal state).

        """

        # Pull out the time and the appropriate proposal.
        if self.n_proposals == 1:
            params_at_t = jax.tree_map(lambda args: args[0], params)
        elif self.n_proposals == 2:
            params_at_t = jax.lax.cond(t == 0,
                                       lambda *_: jax.tree_map(lambda args: args[0], params),
                                       lambda *_: jax.tree_map(lambda args: args[1], params),
                                       None)
        else:
            params_at_t = jax.tree_map(lambda args: args[t], params)

        proposal_inputs = self._proposal_input_generator(dataset, model, particles, t, p_dist, q_state, *inputs)
        q_dist = self.proposal.apply(params_at_t, proposal_inputs)

        return q_dist, None

    def _proposal_input_generator(self, dataset, model, particles, t, p_dist, q_state, *inputs):
        r"""
        Converts inputs of the form (dataset, model, particle[SINGLE], t, p_dist, q_state, *inputs) into a vector object that
        can be input directly into the proposal.

        Args:
            dataset:
            model:
            particles:
            t:
            p_dist:
            q_state:
            *inputs:

        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into proposal.

        """

        # This proposal gets the entire dataset and the current particles.
        proposal_inputs = (dataset, particles)

        model_latent_shape = (model.latent_dim, )

        is_batched = (model_latent_shape != particles.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)


class IGSingleObsProposal(IndependentGaussianProposal):
    """
    IndependentGaussianProposal but where only a single observation at time t is used.
    """

    def _proposal_input_generator(self, dataset, model, particles, t, p_dist, q_state, *inputs):
        assert self.proposal_window_length == 1, "ERROR: Must have a single-length window."

        # This proposal gets the single datapoint and the current particles.
        proposal_inputs = (jax.lax.dynamic_index_in_dim(dataset, t), particles)

        model_latent_shape = (model.latent_dim, )

        is_batched = (model_latent_shape != particles.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)


class IGWindowProposal(IndependentGaussianProposal):
    """
    IndependentGaussianProposal where a window of observations is used.
    """

    # We need to define the method for generating the inputs.
    def _proposal_input_generator(self, dataset, model, particles, t, p_dist, q_state, *inputs):

        masked_idx = np.arange(self.proposal_window_length)
        to_insert = (t + 1 + masked_idx < len(dataset))  # We will insert where the window is inside the dataset.

        # Zero out the elements outside of the valid range.
        clipped_dataset = jax.lax.dynamic_slice(dataset,
                                                (t, *tuple(0 * _d for _d in dataset.shape[1:])),  # NOTE - removed t+1.
                                                (self.proposal_window_length, *dataset.shape[1:]))
        masked_dataset = clipped_dataset * np.expand_dims(to_insert.astype(np.int32), 1)

        # We will pass in whole data into the tilt and then filter out as required.
        proposal_inputs = (masked_dataset, particles)

        model_latent_shape = (model.latent_dim, )

        is_batched = (model_latent_shape != particles.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)


def rebuild_proposal(proposal, env):
    """
    Function that produces another function that wraps the proposal.  This needs wrapping because we define a
    proposal as a function that takes just the inputs (as opposed to the inputs and the parameters of the proposal).
    Therefore, this function partly just closes over the value of the proposal parameters, returning a function with
    a call to these values baked in.

    The proposal may also be parameterized in a funny way, and so this function provides some flexibility in how the
    function is defined and used.

    This is partly to separate out the code as much as possible, but also because vmapping over distribution and
    model objects was proving to be a pain, so this allows you to vmap the proposal inside the proposal itself,
    and then get the results of that and use them as required (i.e. for implementing the ResQ proposal).

    NOTE - both of the proposal functions also return a None, as there is no q_state to pass along.

    Args:
        proposal:               Proposal object.  Will wrap a call to the `.apply` method.
        env:

    Returns:
        (Callable):             Function that can be called as fn(inputs).

    """

    def _rebuild_proposal(_param_vals, _dataset, _model, _encoded_data=None):

        # If there is no proposal, then there is no structure to define.
        if (proposal is None) or (env.config.proposal_structure == 'BOOTSTRAP') or (env.config.proposal_structure == 'NONE'):
            return None

        # We fork depending on the proposal type.
        # Proposal takes arguments of (dataset, model, particles, time, p_dist, q_state, ...).
        if env.config.proposal_structure == 'DIRECT':

            def _proposal(particles, t, p_dist, q_state, *inputs):
                z_dist, new_q_state = proposal.apply(_param_vals, _dataset, _model, particles, t, p_dist, q_state, *inputs)
                return z_dist, new_q_state

        elif env.config.proposal_structure == 'RESQ':

            def _proposal(particles, t, p_dist, q_state, *inputs):
                q_dist, new_q_state = proposal.apply(_param_vals, _dataset, _model, particles, t, p_dist, q_state, *inputs)
                z_dist = tfd.MultivariateNormalFullCovariance(loc=p_dist.mean() + q_dist.mean(),
                                                              covariance_matrix=q_dist.covariance())
                return z_dist, new_q_state

        elif env.config.proposal_structure == 'VRNN_FILTERING_RESQ':

            assert 'Filtering' in str(type(proposal)), "Must use filtering proposal."  # TODO - this is kinda grotty.

            def _proposal(particles, t, p_dist, q_state, *inputs):
                """
                Note that because the VRNN has a deterministic element in the proposal, we need to separate and then re-build the deterministic
                element separately from the continuous/Gaussian RESQ part of the proposal.
                """
                # Pull out the deterministic part of the latent state.
                vrnn_h_dist = p_dist._model[0]
                vrnn_z_dist = p_dist._model[1]

                # NOTE - the VRNN proposal only produces the stochastic part of the state.
                q_z_dist, new_q_state = proposal.apply(_param_vals, _dataset, _model, particles, t, p_dist, q_state, *inputs)

                # Build the proposal part of the state.
                rnn_z_dist = tfd.MultivariateNormalFullCovariance(loc=vrnn_z_dist.mean() + q_z_dist.mean(),
                                                                  covariance_matrix=q_z_dist.covariance())

                # Recapitulate the whole state distribution.
                z_dist = p_dist.__class__((vrnn_h_dist, rnn_z_dist))

                return z_dist, new_q_state

        elif env.config.proposal_structure == 'VRNN_SMOOTHING_RESQ':

            assert 'Smoothing' in str(type(proposal)), "Must use smoothing proposal."  # TODO - this is kinda grotty.
            assert _encoded_data is not None, "Must supple encoded data."

            # assert type(_encoded_data) is tuple, "Encoded data must be a tuple: (forward encoding, backward encoding)."
            # assert _encoded_data[0].shape == _encoded_data[1].shape, "Encoded data shapes are not equal."
            # assert _encoded_data[0].shape[0] == _dataset.shape[0], "Encoded data must be same length as raw data."

            # Now build up the proposal function to directly pass in this processed stuff as input.
            def _proposal(particles, t, p_dist, q_state, *_):
                """
                Note that because the VRNN has a deterministic element in the proposal, we need to separate and then re-build the deterministic
                element separately from the continuous/Gaussian RESQ part of the proposal.

                Note that the inputs are not used here -- they are replaced with an encoded observation.
                """
                # Pull out the deterministic part of the latent state.
                vrnn_h_dist = p_dist._model[0]
                vrnn_z_dist = p_dist._model[1]

                # Grab the encoded state.
                encoded_data_at_t = jax.tree_map(lambda _d: jax.lax.dynamic_index_in_dim(_d, t, keepdims=False), _encoded_data)

                # Build the input tuple.
                inputs_at_t = (encoded_data_at_t, )

                # NOTE - the VRNN proposal only produces the stochastic part of the state.
                q_z_dist, new_q_state = proposal.apply(_param_vals, _dataset, _model, particles, t, p_dist, q_state, *inputs_at_t)

                # Build the proposal part of the state.
                rnn_z_dist = tfd.MultivariateNormalFullCovariance(loc=vrnn_z_dist.mean() + q_z_dist.mean(),
                                                                  covariance_matrix=q_z_dist.covariance())

                # Recapitulate the whole state distribution.
                z_dist = p_dist.__class__((vrnn_h_dist, rnn_z_dist))

                return z_dist, new_q_state

        else:
            raise NotImplementedError()

        return _proposal

    return _rebuild_proposal
