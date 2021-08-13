import jax.numpy as jnp
import jax.scipy.special as spsp
from jax import lax, value_and_grad, jit


def hmm_log_normalizer(log_initial_distn, log_transition_matrix, log_likelihoods):
    def marginalize(alpha, ll):
        alpha = spsp.logsumexp(alpha + ll + log_transition_matrix.T, axis=1)
        return alpha, alpha

    # alphas are in log space and cumulative
    alpha_T, _ = lax.scan(marginalize, log_initial_distn, log_likelihoods[:-1])

    # account for last time step when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1])


hmm_expected_states = value_and_grad(hmm_log_normalizer, argnums=(0, 1, 2))
