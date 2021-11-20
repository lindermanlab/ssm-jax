"""
TODO: update this test to work with new modules
"""

import jax.numpy as np
import jax.random as jr

from tensorflow_probability.substrates import jax as tfp

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.distributions.linreg import GaussianLinearRegression
from ssm.lds import GaussianLDS
from ssm.utils import random_rotation


def random_lds(key, T=10, D=2, N=4):
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


def test_mean_to_h(key, T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2, k3 = jr.split(key, 3)
    lds = random_lds(k1, T=T, D=D, N=N)
    data = jr.normal(k2, shape=(T, N))
    J_diag, J_lower_diag, h = lds.e_step(data)
    big_J, big_h = make_big_Jh(J_diag, J_lower_diag, h)

    mean = jr.normal(k3, shape=(T, D))

    # Compute the linear potential naively
    linear_potential_comp = (big_J @ mean.ravel()).reshape((T, D))

    # Compute with block tridiagonal math
    linear_potential = np.einsum('tij,tj->ti', J_diag, mean)

    # linear_potential[:-1] += np.einsum('tji,tj->ti', precision_lower_diag_blocks, mean[1:])
    linear_potential = linear_potential.at[:-1].add(
        np.einsum('tji,tj->ti', J_lower_diag, mean[1:]))

    # linear_potential[1:] += np.einsum('tij,tj->ti', precision_lower_diag_blocks, mean[:-1])
    linear_potential = linear_potential.at[1:].add(
        np.einsum('tij,tj->ti', J_lower_diag, mean[:-1]))

    assert np.allclose(linear_potential, linear_potential_comp)


def test_mvn_log_prob(key, T=10, D=2, N=4):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2, k3 = jr.split(key, 3)
    lds = random_lds(k1, T=T, D=D, N=N)
    x, y = lds.sample(k2, T)
    J_diag, J_lower_diag, h = lds.natural_parameters(y)
    big_J, big_h = make_big_Jh(J_diag, J_lower_diag, h)

    # Make normal (mean) parameters
    big_Sigma = np.linalg.inv(big_J)
    big_mu = big_Sigma @ big_h
    big_x = x.ravel()
    big_mvn = tfp.distributions.MultivariateNormalFullCovariance(big_mu, big_Sigma)
    lp_comp = big_mvn.log_prob(big_x)

    mvn = MultivariateNormalBlockTridiag(J_diag, J_lower_diag, h)
    lp = mvn.log_prob(x)

    assert np.isclose(lp, lp_comp)
    assert np.isfinite(lp)


def test_lds_log_prob(key, T=10, D=2, N=4):
    from ssm.inference.lds import _exact_e_step, _exact_marginal_likelihood

    k1, k2, k3 = jr.split(key, 3)
    lds = random_lds(k1, T=T, D=D, N=N)
    x, y = lds.sample(k2, T)
    posterior = _exact_e_step(lds, y)

    # Comparison: manually compute the "log c" correction to the posterior log normalizer
    m1 = lds._initial_distribution.mean()
    Q1 = lds._initial_distribution.covariance()
    b = lds._dynamics_distribution.bias
    Q = lds._dynamics_distribution.scale
    d = lds._emissions_distribution.bias
    R = lds._emissions_distribution.scale

    logc = -0.5 * T * D * np.log(2 * np.pi)
    logc += -0.5 * np.linalg.slogdet(Q1)[1]
    logc += -0.5 * np.dot(m1, np.linalg.solve(Q1, m1))
    logc += -0.5 * (T - 1) * np.linalg.slogdet(Q)[1]
    logc += -0.5 * (T - 1) * np.dot(b, np.linalg.solve(Q, b))
    logc += -0.5 * T * N * np.log(2 * np.pi)
    logc += -0.5 * T * np.linalg.slogdet(R)[1]
    logc += -0.5 * np.sum((y - d) * np.linalg.solve(R, (y - d).T).T)

    lp_comp = posterior.log_normalizer + logc
    lp = _exact_marginal_likelihood(lds, y)

    assert np.isclose(lp, lp_comp)
    assert np.isfinite(lp)


def test_lds_laplace_em_hessian(key, T=10, D=2, N=4):
    from ssm.inference.lds import _compute_laplace_precision_blocks

    k1, k2 = jr.split(key, 2)
    lds = random_lds(k1, T=T, D=D, N=N)
    x, y = lds.sample(k2, T)

    J_diag_comp, J_lower_diag_comp, _ = lds.natural_parameters(y)
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(lds, x, y)

    assert np.allclose(J_diag, J_diag_comp)
    assert np.allclose(J_lower_diag, J_lower_diag_comp)


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    test_mean_to_h(key)
    test_mvn_log_prob(key)
    test_lds_log_prob(key)
    test_lds_laplace_em_hessian(key)
