"""
TODO: organize HMM-specific EM functions elsewhere

IDEA:
    - decompose this file into `message_passing.py` routines
"""
from collections import namedtuple
from textwrap import dedent

import jax.numpy as np
from jax import jit, lax, value_and_grad

from functools import partial

import jax.scipy.special as spsp
import jax.random as npr
import jax.experimental.optimizers as optimizers

# from ssm.models.hmm import HMM
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples


### Core message passing routines
def hmm_log_normalizer(log_initial_state_probs, log_transition_matrix, log_likelihoods):
    """
    Compute the marginal likelihood (i.e. log normalizer) under
    a Hidden Markov Model (HMM) with the specified natural parameters.
    
    The normalizer is computed via the forward message passing recursion,

    .. math::
        \log \\alpha_{t+1,j} =
            \log \sum_i \exp \{ \log \\alpha_{t,i} +
                                \log p(x_t | z_t = i) +
                                \log p(z_{t+1} = j | z_t = i) \}

    where, 
    
    .. math::
        \\alpha_{t+1} \propto p(z_{t+1} | x_{1:t}).

    Args:
        log_initial_state_probs: Log of the initial state probabilities.
            A shape ``(K,)`` array where ``K`` is the number of states.

        log_transition_matrix: Log transition matrix or matrices. The shape
            must either be ``(K, K)`` where ``K`` is the number of states
            or ``(T-1, K, K)`` where ``T`` is the length of the sequence.
            In the former case, the entry ``[i,j]`` specifies the log
            probability of transitioning from state ``i`` to state ``j``.
            In the latter case, the ``[t,i,j]`` entry gives the log probability
            of transitioning from state ``z_t = i`` to state ``z_{t+1} = j``
            for ``t=1,...,T-1``.

        log_likelihoods: Log likelihoods combined in a shape ``(T, K)``
            array where ``T`` is the length of the sequence and ``K`` is
            the number of states.  The ``[t, i]`` entry specifies the log
            likelihood of observation ``x[t]`` given state ``z[t] = i``.

    Returns:
        The log probability of the sequence, summing out the discrete states.
    """
    assert log_initial_state_probs.ndim == 1 and log_likelihoods.ndim == 2
    num_states = len(log_initial_state_probs)
    num_timesteps = len(log_likelihoods)
    assert log_likelihoods.shape[1] == num_states

    if log_transition_matrix.ndim == 2:
        # Stationary (fixed) transition probabilities
        assert log_transition_matrix.shape == (num_states, num_states)
        return _stationary_hmm_log_normalizer(
            log_initial_state_probs, log_transition_matrix, log_likelihoods
        )

    elif log_transition_matrix.ndim == 3:
        # Time-varying transition probabilities
        assert log_transition_matrix.shape == (
            num_timesteps - 1,
            num_states,
            num_states,
        )
        return _nonstationary_hmm_log_normalizer(
            log_initial_state_probs, log_transition_matrix, log_likelihoods
        )

    else:
        raise Exception("`log_transition_matrix` must be either 2d or 3d.")


def _stationary_hmm_log_normalizer(
    log_initial_state_probs, log_transition_matrix, log_likelihoods
):
    def marginalize(alpha, ll):
        alpha = spsp.logsumexp(alpha + ll + log_transition_matrix.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        marginalize, log_initial_state_probs, log_likelihoods[:-1]
    )

    # Note: alphas is interesting too, for forward filtering
    # Could return ..., np.row_stack((log_initial_state_probs, alphas))

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1])


def _nonstationary_hmm_log_normalizer(
    log_initial_state_probs, log_transition_matrices, log_likelihoods
):
    def marginalize(alpha, prms):
        log_P, ll = prms
        alpha = spsp.logsumexp(alpha + ll + log_P.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        marginalize,
        log_initial_state_probs,
        (log_transition_matrices, log_likelihoods[:-1]),
    )

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1])

hmm_expected_states = jit(value_and_grad(hmm_log_normalizer, argnums=(0, 1, 2)))
hmm_expected_states.__doc__ = dedent(   
    """
    Compute posterior expectations of the latent states in a Hidden Markov
    Model (HMM) with the specified natural parameters. 
    
    The expectations are computed using a special property of exponential 
    family distributions: the expected sufficient statistics are equal to 
    the gradient of the log normalizer with respect to the natural parameters.

    For an HMM, the sufficient statistics are simply indicator functions

    .. math::

            t(z)_1 &= [\mathbb{I}[z_1 = 1], \ldots, \mathbb{I}[z_1=K]] \\\\
            t(z)_2 &= [\mathbb{I}[z_t = 1], \ldots, \mathbb{I}[z_t=K]] \\text{ for } t = 1, ..., T \\\\
            t(z)_3 &= [\mathbb{I}[z_t = k_1, z_{t+1} = k_2]] \\text{ for } t = 1, ..., T-1; k_1 = 1,...,K; k_2 = 1, ..., K
        
    The expected sufficient statistics are the probabilities of these
    events under the posterior distribution determined by the natural parameters,
    i.e. arguments to this function.

    Args:
        log_initial_distn: Log of the initial state distribution.
            A shape ``(K,)`` array where ``K`` is the number of states.
        log_transition_matrix: Log transition matrix or matrices. The shape
            must either be ``(K, K)`` where ``K`` is the number of states
            or ``(T-1, K, K)`` where ``T`` is the length of the sequence.
            In the former case, the entry ``[i,j]`` specifies the log
            probability of transitioning from state ``i`` to state ``j``.
            In the latter case, the ``[t,i,j]`` entry gives the log probability
            of transitioning from state ``z_t = i`` to state ``z_{t+1} = j``
            for ``t=1,...,T-1``.
        log_likelihoods: Log likelihoods combined in a shape ``(T, K)``
            array where ``T`` is the length of the sequence and ``K`` is
            the number of states.  The ``[t, i]`` entry specifies the log
            likelihood of observation ``x[t]`` given state ``z[t] = i``.

    Returns:
        The log normalizer and a tuple including of expected sufficient statistics.
    """
)
    