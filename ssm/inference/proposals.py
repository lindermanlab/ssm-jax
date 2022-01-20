"""
Proposal templates for SMC (+FIVO).
"""
import jax
import jax.numpy as np
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from copy import deepcopy as dc
import flax.linen as nn

# Import some ssm stuff.
from ssm.inference.conditional_generators import IndependentGaussianGenerator
import ssm.nn_util as nn_util


class IndependentGaussianProposal:
    """
    Define a proposal FUNCTION that is an indpendent gaussian.  This Module is actually just a thin wrapper over a set
    of Linen modules.

    To define a different proposal FUNCTION here (as opposed to a proposal structure), change this class.

    This modules `.apply` method also wraps a call to the input generator, which takes the standard proposal
    parameterization (dataset, model, particles, t, p_dist, q_state) and flattens it into the right form.

    Args:
        - n_proposals (int):
            Number of independent proposals to define.  Here must be equal to one (other definitions may use multiple
            proposals independently indexed by discrete state/time etc.

        - stock_proposal_input_without_q_state (tuple):
            Tuple of the stock proposal input used in SMC, without the `q_state` (as this state must be defined here).
            Of type: (dataset, model, particles, time, p_dist).

        - dummy_output (ndarray):
            Correctly shaped ndarray of dummy values used to define the output layer.

    Returns:
        - None

    """

    def __init__(self, n_proposals, stock_proposal_input, dummy_output,
                 trunk_fn=None, head_mean_fn=None, head_log_var_fn=None, proposal_window_length=None):

        # Work out the number of proposals.
        assert (n_proposals == 1) or (n_proposals == 2) or (n_proposals == len(stock_proposal_input[0])), \
            'Can only use a single proposal, two proposals (init and single), or as many proposals as there are states.'
        self.n_proposals = n_proposals

        self.proposal_window_length = proposal_window_length

        # Re-build the full input that will be provided.
        full_input = stock_proposal_input
        self._dummy_processed_input = self._proposal_input_generator(*full_input)[0]
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
            - key (jax.PRNGKey):    Seed to seed initialization.

        Returns:
            - parameters:           FrozenDict of the parameters of the initialized proposal.

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
            # If there are two proposals then assume that one is the initial proposal and the other is the static proposal.
            params_at_t = jax.lax.cond(t == 0,
                                       lambda *_: jax.tree_map(lambda args: args[0], params),
                                       lambda *_: jax.tree_map(lambda args: args[1], params),
                                       None
                                       )
        else:
            params_at_t = jax.tree_map(lambda args: args[t], params)

        proposal_inputs = self._proposal_input_generator(dataset, model, particles, t, p_dist, q_state, *inputs)
        q_dist = self.proposal.apply(params_at_t, proposal_inputs)

        # # TODO - Can force the optimal proposal here for the default GDM example..
        # _prop_inp_old = proposal_inputs
        # if proposal_inputs.ndim == 1:
        #     proposal_inputs = np.expand_dims(proposal_inputs, axis=0)
        #
        # mean = jax.lax.cond(
        #          t == 0,
        #          lambda *args: (proposal_inputs[..., 0] / (np.asarray([9.0 + 1.0]))),
        #          lambda *args: (((9.0 - np.asarray([t]) + 1) * proposal_inputs[:, 1]) + proposal_inputs[:, 0]) / (9.0 - np.asarray([t]) + 1 + 1),
        #          None)
        #
        # std = jax.lax.cond(t == 0,
        #                    lambda *args: (mean * 0.0) + np.sqrt(10.0 / 11.0 + np.asarray([0])),
        #                    lambda *args: (mean * 0.0) + np.sqrt(1.0 / (1.0 + (1.0 / (1.0 + ((9.0 - np.asarray([t])) * 1.0))))),
        #                    None)
        #
        # # q_dist = tfd.MultivariateNormalDiag((q_dist.mean().squeeze() * 0.0) + np.expand_dims(mean, -1), (q_dist.stddev() * 0.0) + std)
        # if _prop_inp_old.ndim == 1:
        #     q_dist = tfd.MultivariateNormalDiag(mean, std)
        # else:
        #     q_dist = tfd.MultivariateNormalDiag(np.expand_dims(mean, axis=1), np.expand_dims(std, axis=1))
        # # TODO - Can force the optimal proposal here for the default GDM example..

        return q_dist, None

    def _proposal_input_generator(self, _dataset, _model, _particles, _t, _p_dist, _q_state, *_inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t, p_dist, q_state) into a vector object that
        can be input into the proposal.

        Args:


        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into proposal.

        """

        # This proposal gets the entire dataset and the current particles.
        _proposal_inputs = (_dataset, _particles)

        _model_latent_shape = (_model.latent_dim, )

        _is_batched = (_model_latent_shape != _particles.shape)
        if not _is_batched:
            return nn_util.vectorize_pytree(_proposal_inputs)
        else:
            _vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return _vmapped(*_proposal_inputs)


class IGPerStepProposal(IndependentGaussianProposal):
    """

    """
    pass


class IGSingleObsProposal(IndependentGaussianProposal):

    def _proposal_input_generator(self, _dataset, _model, _particles, _t, _p_dist, _q_state, *_inputs):
        """

        """

        # This proposal gets the single datapoint and the current particles.
        _proposal_inputs = (jax.lax.dynamic_index_in_dim(_dataset, _t), _particles)

        _model_latent_shape = (_model.latent_dim, )

        _is_batched = (_model_latent_shape != _particles.shape)
        if not _is_batched:
            return nn_util.vectorize_pytree(_proposal_inputs)
        else:
            _vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return _vmapped(*_proposal_inputs)


class IGWindowProposal(IndependentGaussianProposal):

    # window_length = 2

    # We need to define the method for generating the inputs.
    def _proposal_input_generator(self, _dataset, _model, _particles, _t, _p_dist, _q_state, *_inputs):
        """

        """

        _masked_idx = np.arange(self.proposal_window_length)
        _to_insert = (_t + 1 + _masked_idx < len(_dataset))  # We will insert where the window is inside the dataset.

        # Zero out the elements outside of the valid range.
        _clipped_dataset = jax.lax.dynamic_slice(_dataset,
                                                (_t+1, *tuple(0 * _d for _d in _dataset.shape[1:])),
                                                (self.proposal_window_length, *_dataset.shape[1:]))
        _masked_dataset = _clipped_dataset * np.expand_dims(_to_insert.astype(np.int32), 1)

        # We will pass in whole data into the tilt and then filter out as required.
        _proposal_inputs = (_masked_dataset, _particles)

        _model_latent_shape = (_model.latent_dim, )

        _is_batched = (_model_latent_shape != _particles.shape)
        if not _is_batched:
            return nn_util.vectorize_pytree(_proposal_inputs)
        else:
            _vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return _vmapped(*_proposal_inputs)


def rebuild_proposal(proposal, proposal_structure):
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
        proposal_structure:     String indicating the type of proposal structure to use.

    Returns:
        (Callable):             Function that can be called as fn(inputs).

    """

    def _rebuild_proposal(_param_vals, _dataset, _model, ):

        # If there is no proposal, then there is no structure to define.
        if (proposal is None) or (proposal_structure == 'BOOTSTRAP') or (proposal_structure == 'NONE'):
            return None

        # We fork depending on the proposal type.
        # Proposal takes arguments of (dataset, model, particles, time, p_dist, q_state, ...).
        if proposal_structure == 'DIRECT':

            def _proposal(particles, t, p_dist, q_state, *inputs):
                z_dist, new_q_state = proposal.apply(_param_vals, _dataset, _model, particles, t, p_dist, q_state, *inputs)
                return z_dist, new_q_state

        elif proposal_structure == 'RESQ':

            def _proposal(particles, t, p_dist, q_state, *inputs):
                q_dist, new_q_state = proposal.apply(_param_vals, _dataset, _model, particles, t, p_dist, q_state, *inputs)
                z_dist = tfd.MultivariateNormalFullCovariance(loc=p_dist.mean() + q_dist.mean(),
                                                              covariance_matrix=q_dist.covariance())
                return z_dist, new_q_state

        elif proposal_structure == 'VRNN':

            def _proposal(particles, t, p_dist, q_state, *inputs):
                raise NotImplementedError()

        else:
            raise NotImplementedError()

        return _proposal

    return _rebuild_proposal
