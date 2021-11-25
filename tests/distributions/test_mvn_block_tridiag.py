import pytest

import jax.numpy as np
import jax.random as jr

from tensorflow_probability.substrates import jax as tfp

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.distributions.linreg import GaussianLinearRegression
from ssm.lds import GaussianLDS
from ssm.utils import random_rotation

SEED = jr.PRNGKey(0)

def random_lds(key, D=2, N=4):
    """
    Make a random LDS with T time steps, D latent dimensions, N emission dims.
    """
    return GaussianLDS(D, N, seed=key)


def make_big_Jh(J_diag, J_lower_diag, h):
    T, D = h.shape
    big_J = np.zeros((T * D, T * D))
    for t in range(T):
        tslc = slice(t * D, (t+1) * D)
        big_J = big_J.at[tslc, tslc].add(J_diag[t])

    for t in range(T-1):
        tslc = slice(t * D, (t+1) * D)
        tp1slc = slice((t+1) * D, (t+2) * D)
        big_J = big_J.at[tp1slc, tslc].add(J_lower_diag[t])
        big_J = big_J.at[tslc, tp1slc].add(J_lower_diag[t].T)

    big_h = h.ravel()

    return big_J, big_h


def test_mean_to_h(T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2, k3 = jr.split(SEED, 3)
    lds = random_lds(k1, D=D, N=N)
    data = jr.normal(k2, shape=(T, N))
    posterior = lds.e_step(data)

    J_diag = posterior.precision_diag_blocks
    J_lower_diag = posterior.precision_lower_diag_blocks
    h = posterior.linear_potential
    big_J, _ = make_big_Jh(J_diag, J_lower_diag, h)

    mean = jr.normal(k3, shape=(T, D))

    # Compute the linear potential naively
    linear_potential_comp = (big_J @ mean.ravel()).reshape((T, D))

    # Compute with block tridiagonal math
    linear_potential = np.einsum('tij,tj->ti', J_diag, mean)
    linear_potential = linear_potential.at[:-1].add(
        np.einsum('tji,tj->ti', J_lower_diag, mean[1:]))
    linear_potential = linear_potential.at[1:].add(
        np.einsum('tij,tj->ti', J_lower_diag, mean[:-1]))

    assert np.allclose(linear_potential, linear_potential_comp)


def test_mvn_log_prob(T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2 = jr.split(SEED, 2)
    lds = random_lds(k1, D=D, N=N)
    x, y = lds.sample(k2, T)
    posterior = lds.e_step(y)

    J_diag = posterior.precision_diag_blocks
    J_lower_diag = posterior.precision_lower_diag_blocks
    h = posterior.linear_potential
    big_J, big_h = make_big_Jh(J_diag, J_lower_diag, h)

    # Make normal (mean) parameters
    big_Sigma = np.linalg.inv(big_J)
    big_mu = big_Sigma @ big_h
    big_x = x.ravel()
    big_mvn = tfp.distributions.MultivariateNormalFullCovariance(big_mu, big_Sigma)
    lp_comp = big_mvn.log_prob(big_x)

    lp = posterior.log_prob(x)

    assert np.isclose(lp, lp_comp, atol=1e-4)
    assert np.isfinite(lp)


def test_lds_log_prob(T=10, D=2, N=4):
    k1, k2 = jr.split(SEED, 2)
    lds = random_lds(k1, D=D, N=N)
    x, y = lds.sample(k2, T)
    posterior = lds.e_step(y)

    # Comparison: manually compute the "log c" correction to the posterior log normalizer
    m1 = lds._initial_condition._distribution.mean()
    Q1 = lds._initial_condition._distribution.covariance()
    b = lds._dynamics._distribution.bias
    Q = lds._dynamics._distribution.covariance
    d = lds._emissions._distribution.bias
    R = lds._emissions._distribution.covariance

    logc = -0.5 * T * D * np.log(2 * np.pi)
    logc += -0.5 * np.linalg.slogdet(Q1)[1]
    logc += -0.5 * np.dot(m1, np.linalg.solve(Q1, m1))
    logc += -0.5 * (T - 1) * np.linalg.slogdet(Q)[1]
    logc += -0.5 * (T - 1) * np.dot(b, np.linalg.solve(Q, b))
    logc += -0.5 * T * N * np.log(2 * np.pi)
    logc += -0.5 * T * np.linalg.slogdet(R)[1]
    logc += -0.5 * np.sum((y - d) * np.linalg.solve(R, (y - d).T).T)

    lp_comp = posterior.log_normalizer + logc
    lp = lds.log_probability(x, y) - posterior.log_prob(x)

    assert np.isclose(lp, lp_comp, atol=1e-4)
    assert np.isfinite(lp)


def test_mvn_expected_states(T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2 = jr.split(SEED, 2)
    lds = random_lds(k1, D=D, N=N)
    x, y = lds.sample(k2, T)
    posterior = lds.e_step(y)
    expected_states = posterior.expected_states

    J_diag = posterior.precision_diag_blocks
    J_lower_diag = posterior.precision_lower_diag_blocks
    h = posterior.linear_potential
    big_J, big_h = make_big_Jh(J_diag, J_lower_diag, h)
    expected_states_comp = np.linalg.solve(big_J, big_h).reshape(T, D)

    assert np.allclose(expected_states, expected_states_comp, atol=1e-4)
    assert np.all(np.isfinite(expected_states))


def test_mvn_expected_states_squared(T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2 = jr.split(SEED, 2)
    lds = random_lds(k1, D=D, N=N)
    x, y = lds.sample(k2, T)
    posterior = lds.e_step(y)
    ExxT = posterior.expected_states_squared

    J_diag = posterior.precision_diag_blocks
    J_lower_diag = posterior.precision_lower_diag_blocks
    h = posterior.linear_potential
    big_J, big_h = make_big_Jh(J_diag, J_lower_diag, h)
    mean = np.linalg.solve(big_J, big_h).reshape(T, D)
    cov = np.linalg.inv(big_J).reshape(T, D, T, D)
    ExxT_comp = cov[np.arange(T), :, np.arange(T), :] + np.einsum('ti,tj->tij', mean, mean)

    assert np.allclose(ExxT, ExxT_comp, atol=1e-4)
    assert np.all(np.isfinite(mean))


def test_mvn_expected_states_next_states(T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2 = jr.split(SEED, 2)
    lds = random_lds(k1, D=D, N=N)
    x, y = lds.sample(k2, T)
    posterior = lds.e_step(y)
    ExnxT = posterior.expected_states_next_states

    J_diag = posterior.precision_diag_blocks
    J_lower_diag = posterior.precision_lower_diag_blocks
    h = posterior.linear_potential
    big_J, big_h = make_big_Jh(J_diag, J_lower_diag, h)
    mean = np.linalg.solve(big_J, big_h).reshape(T, D)
    cov = np.linalg.inv(big_J).reshape(T, D, T, D)
    ExnxT_comp = cov[np.arange(1, T), :, np.arange(T-1), :] + np.einsum('ti,tj->tij', mean[1:], mean[:-1])

    assert np.allclose(ExnxT, ExnxT_comp, atol=1e-4)
    assert np.all(np.isfinite(mean))


def test_lds_laplace_em_hessian(T=10, D=2, N=4):
    from ssm.inference.laplace_em import _compute_laplace_precision_blocks

    k1, k2 = jr.split(SEED, 2)
    lds = random_lds(k1, D=D, N=N)
    x, y = lds.sample(k2, T)

    # shorthand LDS parameters
    # m1 = lds.initial_mean
    # Q1 = lds.initial_covariance
    # A = lds.dynamics_matrix
    # b = lds.dynamics_bias
    # Q = lds.dynamics_noise_covariance
    # C = lds.emissions_matrix
    # d = lds.emissions_bias
    # R = lds.emissions_noise_covariance

    # from jax import vmap, hessian, jacfwd, jacrev
    # # initial
    # J_init = -1 * hessian(lds.initial_distribution().log_prob)(x[0])
    # assert np.allclose(J_init, np.linalg.inv(Q1), atol=1e-6)
    # # dynamics
    # f = lambda xt, xtp1: lds.dynamics_distribution(xt).log_prob(xtp1)
    # J_11 = -1 * vmap(hessian(f, argnums=0))(x[:-1], x[1:])
    # assert np.allclose(J_11, np.dot(A.T, np.linalg.solve(Q, A)), atol=1e-6)
    # J_22 = -1 * vmap(hessian(f, argnums=1))(x[:-1], x[1:])
    # assert np.allclose(J_22, np.linalg.inv(Q), atol=1e-6)
    # J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=1), argnums=0))(x[:-1], x[1:])
    # assert np.allclose(J_21, -np.linalg.solve(Q, A), atol=1e-6)
    # # emissions
    # f = lambda x, y: lds.emissions_distribution(x).log_prob(y)
    # J_obs = -1 * vmap(hessian(f, argnums=0))(x, y)
    # assert np.allclose(J_obs, np.dot(C.T, np.linalg.solve(R, C)), atol=1e-6)

    # Now check
    posterior = lds.e_step(y)
    J_diag_comp = posterior.precision_diag_blocks
    J_lower_diag_comp = posterior.precision_lower_diag_blocks
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(lds, x, y)

    assert np.allclose(J_diag, J_diag_comp)
    assert np.allclose(J_lower_diag, J_lower_diag_comp)


