from collections import namedtuple
from functools import partial

import jax.numpy as np
from jax import jit, value_and_grad
from jax.scipy.linalg import solve_triangular
from jax import lax

from ssm.models.lds import GaussianLDS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples


# TODO @slinderman @schlagercollin: Consider data classes
LDSPosterior = namedtuple(
    "LDSPosterior", ["marginal_likelihood",
                     "expected_states",
                     "expected_states_squared",
                     "expected_transitions"]
)



def lds_log_normalizer(J_diag, J_lower_diag, h, logc):
    seq_len, dim, _ = J_diag.shape
    # assert L.shape == (seq_len-1, dim, dim)
    # assert h.shape == (seq_len, dim)

    # Pad the L's with one extra set of zeros for the last predict step
    J_lower_diag_pad = np.concatenate((J_lower_diag, np.zeros((1, dim, dim))), axis=0)

    def marginalize(carry, t):
        Jp, hp, lp = carry

        # Condition
        Jc = J_diag[t] + Jp
        hc = h[t] + hp

        # Predict -- Cholesky approach seems unstable!
        sqrt_Jc = np.linalg.cholesky(Jc)
        trm1 = solve_triangular(sqrt_Jc, hc, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_lower_diag_pad[t].T, lower=True)
        log_Z = 0.5 * dim * np.log(2 * np.pi)
        log_Z += -np.sum(np.log(np.diag(sqrt_Jc)))
        log_Z += 0.5 * np.dot(trm1.T, trm1)
        Jp = -np.dot(trm2.T, trm2)
        hp = -np.dot(trm2.T, trm1)

        # Alternative predict step:
        # log_Z = 0.5 * dim * np.log(2 * np.pi)
        # log_Z += -0.5 * np.linalg.slogdet(Jc)[1]
        # log_Z += 0.5 * np.dot(hc, np.linalg.solve(Jc, hc))
        # Jp = -np.dot(L_pad[t], np.linalg.solve(Jc, L_pad[t].T))
        # hp = -np.dot(L_pad[t], np.linalg.solve(Jc, hc))

        new_carry = Jp, hp, lp + log_Z
        return new_carry, None

    # Initialize
    Jp0 = np.zeros((dim, dim))
    hp0 = np.zeros((dim,))
    (_, _, lp), _ = lax.scan(marginalize, (Jp0, hp0, logc), np.arange(seq_len))
    return lp


# Expectation-Maximization
def _e_step(lds, data):
    marginal_likelihood, (E_neg_half_xxT, E_neg_xnxT, Ex) = \
        value_and_grad(lds_log_normalizer)(*lds.natural_parameters(data))

    ExxT = -2 * E_neg_half_xxT
    ExnxT = -E_neg_xnxT
    return LDSPosterior(marginal_likelihood, Ex, ExxT, ExnxT)


# Solve a linear regression given expected sufficient statistics
def fit_linear_regression(Ex, Ey, ExxT, EyxT, EyyT, En):
    big_ExxT = np.row_stack([np.column_stack([ExxT, Ex]),
                            np.concatenate( [Ex.T, np.array([En])])])
    big_EyxT = np.column_stack([EyxT, Ey])
    Cd = np.linalg.solve(big_ExxT, big_EyxT.T).T
    C, d = Cd[:, :-1], Cd[:, -1]
    R = (EyyT - Cd @ big_EyxT.T - big_EyxT @ Cd.T + Cd @ big_ExxT @ Cd.T) / En
    return C, d, R

def _m_step(lds, data, posterior, prior=None):
    # TODO: update initial distribution
    m1 = np.zeros(Ex.shape[1])
    Q1 = np.eye(Ex.shape[1])

    Ex = posterior.expected_states
    ExxT = posterior.expected_states_squared
    ExnxT = posterior.expected_transitions

    # TODO: use the matrix-normal-inverse-Wishart prior
    A, b, Q = fit_linear_regression(Ex[:-1].sum(axis=0),
                                    Ex[1:].sum(axis=0),
                                    ExxT[:-1].sum(axis=0),
                                    ExnxT.sum(axis=0),
                                    ExxT[1:].sum(axis=0),
                                    len(data) - 1)

    C, d, R = fit_linear_regression(Ex.sum(axis=0),
                                    data.sum(axis=0),
                                    ExxT.sum(axis=0),
                                    data.T.dot(Ex),
                                    data.T.dot(data),
                                    len(data))

    return GaussianLDS(m1, np.linalg.cholesky(Q1),
                       A, b, np.linalg.cholesky(Q),
                       C, d, np.linalg.cholesky(R))


def em(lds,
       data,
       num_iters=100,
       tol=1e-4,
       verbosity=Verbosity.DEBUG,
       m_step_type="exact",
       num_inner=1,
       patience=5,
    ):

    @jit
    def step(lds):
        posterior = _e_step(lds, data)
        if m_step_type == "exact":
            lds = _m_step(lds, data, posterior)

        # elif m_step_type == "sgd_marginal_likelihood":
        #     _generic_m_step(lds, data, posterior, num_iters=num_inner)
        # elif m_step_type == "sgd_expected_log_prob":
        #     _generic_m_step_elbo(lds, data, posterior, num_iters=num_inner)
        # else:
        #     raise ValueError("unrecognized")
        return lds, posterior

    # Run the EM algorithm to convergence
    log_probs = [np.nan]
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, log_probs[-1])
    if verbosity:
        pbar.set_description("[jit compiling...]")
    init_patience = patience
    for itr in pbar:
        lds, posterior = step(lds)
        lp = posterior.marginal_likelihood
        log_probs.append(lp)
        if verbosity:
            pbar.set_description("LP: {:.3f}".format(lp))

        # Check for convergence
        # TODO: make this cleaner with patience
        if abs(log_probs[-1] - log_probs[-2]) < tol and itr > 1:
            if patience == 0:
                if verbosity:
                    pbar.set_description("[converged] LP: {:.3f}".format(lp))
                    pbar.refresh()
                break
            else:
                patience -= 1
        else:
            patience = init_patience

    return np.array(log_probs), lds, posterior