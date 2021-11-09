import jax.numpy as np
import jax.scipy.special as spsp
from jax import lax, value_and_grad, vmap
from jax.tree_util import register_pytree_node_class

from ssm.utils import logspace_tensordot


def _factorial_hmm_log_normalizer(log_initial_state_probs,
                                  log_transition_matrices,
                                  log_likelihoods):
    """
    Computes the log normalization constant of the joint distribution over
    observations and latent state trajectories of a factorial HMM with ``m``
    groups of latent state variables with ``n_1, n_2, ..., n_m`` discrete
    states.

    Parameters
    ----------
    log_initial_state_probs : (n_1, n_2, ..., n_m)-array
        Log unnormalized state probalitities at the initial timestep.

    log_transition_matrices : ((n_1, n_1)-array, (n_2, n_2)-array, ...)
        Log transition matrices for ``m`` groups of variables.

    log_likelihoods : (T, n_1, n_2, ..., n_m)-array
        Log likelihoods of all states over ``T`` timesteps.

    Returns
    -------
    log_normalizer : float
        Log normalization constant.
    """
    # Compute the initial message for all combinations of states
    alpha_0 = log_initial_state_probs[0]
    for lp in log_initial_state_probs[1:]:
        alpha_0 = alpha_0[..., None] + lp

    def marginalize(alphas, log_likes):
        # Weight each state by its log-likelihood to get log p(z_t | x_{1:t})
        alphas += log_likes

        # Matrix-multiply each in transition matrix in log-space
        # to get log p(z_{t+1} | x_{1:t}).
        # Note: may want to lax.fori_loop this step.
        for k, log_trans in enumerate(log_transition_matrices):
            alphas = logspace_tensordot(alphas, log_trans, k)

        return alphas, alphas

    # Forward pass.
    alpha_T, alphas = lax.scan(marginalize, alpha_0, log_likelihoods[:-1])

    # Include the initial potentials to get log Pr(z_t | x_{1:t-1})
    # for all time steps. These are the "filtered potentials".
    filtered_potentials = np.concatenate([alpha_0[None, ...], alphas], axis=0)

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1]), filtered_potentials


@register_pytree_node_class
class FactorialHMMPosterior:
    """
    TODO
    """
    def __init__(self,
                 log_initial_state_probs,
                 log_likelihoods,
                 log_transition_matrices,
                 log_normalizer,
                 filtered_potentials,
                 expected_initial_states,
                 expected_states,
                 expected_transitions
             ) -> None:
        self._log_initial_state_probs = log_initial_state_probs
        self._log_transition_matrices = log_transition_matrices
        self._log_likelihoods = log_likelihoods
        self._log_normalizer = log_normalizer
        self._filtered_potentials = filtered_potentials
        self._expected_initial_states = expected_initial_states
        self._expected_states = expected_states
        self._expected_transitions = expected_transitions

    @classmethod
    def infer(cls,
              log_initial_state_probs,
              log_likelihoods,
              log_transition_matrices):
        """
        Run message passing code to get the log normalizer, the filtered potentials,
        and the expected states and transitions. Then return a posterior with all
        of these parameters cached.

        Note: We assume that the given potentials are for a single time series!
        """
        # Since this is a natural exponential family, the expected states and transitions
        # are given by gradients of the log normalizer.
        f = value_and_grad(_factorial_hmm_log_normalizer, argnums=(0, 1, 2), has_aux=True)
        (log_normalizer, filtered_potentials), \
            (expected_initial_states, expected_transitions, expected_states) = \
                f(log_initial_state_probs, log_transition_matrices, log_likelihoods)

        return cls(log_initial_state_probs,
                   log_likelihoods,
                   log_transition_matrices,
                   log_normalizer,
                   filtered_potentials,
                   expected_initial_states,
                   expected_states,
                   expected_transitions)

    @property
    def log_normalizer(self):
        return self._log_normalizer

    def _mean(self):
        return self._expected_states

    @property
    def expected_initial_states(self):
        return self._expected_initial_states

    @property
    def expected_states(self):
        return self._expected_states

    @property
    def expected_transitions(self):
        return self._expected_transitions

    def tree_flatten(self):
        aux_data = None
        children = (
            self._log_initial_state_probs,
            self._log_transition_matrices,
            self._log_likelihoods,
            self._log_normalizer,
            self._filtered_potentials,
            self._expected_initial_states,
            self._expected_states,
            self._expected_transitions,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __getitem__(self, index):
        print("called __getitem__ on FactorialHMMPosterior")
        return FactorialHMMPosterior(
            self._log_initial_state_probs[index],
            self._log_transition_matrices[index],
            self._log_likelihoods[index],
            self._log_normalizer[index],
            self._filtered_potentials[index],
            self._expected_initial_states[index],
            self._expected_states[index],
            self._expected_transitions[index],
        )
