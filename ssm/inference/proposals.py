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

# Import some ssm stuff.
from ssm.inference.conditional_generators import build_independent_gaussian_generator
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

    def __init__(self, n_proposals, stock_proposal_input_without_q_state, dummy_output):

        assert n_proposals == 1, 'Can only use a single proposal.'

        # Re-build the full input that will be provided.
        q_state = None
        full_input = (*stock_proposal_input_without_q_state, q_state)
        self._dummy_processed_input = self._proposal_input_generator(*full_input)
        output_dim = nn_util.vectorize_pytree(dummy_output).shape[0]

        # Define a more conservative initialization.
        w_init_mean = lambda *args: (0.01 * jax.nn.initializers.normal()(*args))

        trunk_fn = None  # MLP(features=(3, 4, 5), kernel_init=w_init)
        head_mean_fn = nn_util.Static(output_dim, kernel_init=w_init_mean)
        head_log_var_fn = nn_util.Static(output_dim, kernel_init=w_init_mean)

        # Build out the function approximator.
        self.proposal = build_independent_gaussian_generator(self._dummy_processed_input,
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
        return self.proposal.init(key, self._dummy_processed_input)

    def apply(self, params, inputs):
        """

        Args:
            params (FrozenDict):    FrozenDict of the parameters of the proposal.

            inputs (tuple):         Tuple of standard inputs to the proposal in SMC:
                                    (dataset, model, particles, time, p_dist)

        Returns:
            (Tuple): (TFP distribution over latent state, updated q internal state).

        """
        proposal_inputs = self._proposal_input_generator(*inputs)
        q_dist = self.proposal.apply(params, proposal_inputs)
        return q_dist, None

    def _proposal_input_generator(self, *_inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t, p_dist, q_state) into a vector object that
        can be input into the proposal.

        Args:
            *_inputs (tuple):       Tuple of standard inputs to the proposal in SMC:
                                    (dataset, model, particles, time, p_dist)

        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into proposal.

        """

        dataset, _, particles, t, _, _ = _inputs  # NOTE - this part of q can't actually use model or p_dist.
        proposal_inputs = (jax.lax.dynamic_index_in_dim(_inputs[0], index=0, axis=0, keepdims=False), _inputs[2])

        is_batched = (_inputs[1].latent_dim != particles.shape[0])
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)


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

    def _rebuild_proposal(_param_vals):
        # If there is no proposal, then there is no structure to define.
        if (proposal is None) or (proposal_structure == 'BOOTSTRAP'):
            return None

        # We fork depending on the proposal type.
        # Proposal takes arguments of (dataset, model, particles, time, p_dist, q_state, ...).
        if proposal_structure == 'DIRECT':

            def _proposal(*_input):
                z_dist, q_state = proposal.apply(_param_vals, _input)
                return z_dist, q_state

        elif proposal_structure == 'RESQ':

            def _proposal(*_input):
                dataset, model, particles, t, p_dist, q_state = _input
                q_dist, q_state = proposal.apply(_param_vals, _input)
                z_dist = tfd.MultivariateNormalFullCovariance(loc=p_dist.mean() + q_dist.mean(),
                                                              covariance_matrix=q_dist.covariance())
                return z_dist, q_state
        else:
            raise NotImplementedError()

        return _proposal

    return _rebuild_proposal
