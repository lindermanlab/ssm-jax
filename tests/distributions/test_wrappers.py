import pytest

import jax.numpy as np
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd
from ssm.distributions.wrappers import MultivariateNormalFullCovariance

def random_orthogonal_matrix(seed: jr.PRNGKey, N: int):
    """Generate a random orthogonal matrix from the Haar distriubtion.

    Let A be an N x N matrix whose elements :math:`x_{i,j} ~ N(0, \sigma**2)`,
    and let it have a QR factorization :math:`A = QR`, with the factorization
    so that the diagonal elements of :math:`R` are positive. Then, :math:`Q` is
    from a random orthogonal matrix from the Haar distrbution.
    
    The method here requires O(N^3) operations. A more efficient procedure
    involving the successive product of :math:`N-1` Householder transformations
    requires only O(N^2) operations. See reference for more detail.

    References:
    G. W. Stewart. "The efficient generation of random orthogonal
    matrices with an application to condition estimators." Sigam J. Numer Anal.,
    17:3 (1980), pp. 403-409.

    """

    A = jr.normal(seed, (N,N))

    Q, R = np.linalg.qr(A)

    # Normalize the factorization so that diagonal elements of R are positive
    return np.diag(np.sign(np.diag(R))) @ Q

def random_matrix_given_eigvals(seed, eigvals):
    """Generates a random symmetric matrix with specifed eigenvalues.

    The orthogonal matrix Q can be expressed as a product of n-1 Householder matrices
    of rowing effective dimension

    Q is then from the Haar 

    Params
        seed (jr.PRNGKey): random seed
        eigvals (np.ndarray): array of eigenvalues, shape ([B1,...Bb],D,)

    Return
        mat (np.ndarray): shape ([B1,...Bb],D,D)

    """

    D = eigvals.shape[-1]
    Q = random_orthogonal_matrix(seed, D)
    
    return Q.T @ np.diag(eigvals) @ Q
    
def test_mvn_unweighted_mle():
    """
    Test MVN MLE via sufficient statistics function
    """

    seed = iter(jr.split(jr.PRNGKey(1604), 5))

    # Randomly determine MVN parameters
    D = 3
    mu = jr.normal(next(seed), (D,)) * 2

    #- Choose positive eigvals so our matrix is PSD
    eigvals = jr.uniform(next(seed), (D,), minval=0.1, maxval=2)
    Sigma = random_matrix_given_eigvals(next(seed), eigvals)

    assert np.allclose(np.sort(np.linalg.eigvals(Sigma)),
                       np.sort(eigvals))

    distr = MultivariateNormalFullCovariance(mu, Sigma)

    # Generate date from MVN
    N = 1000
    data = distr.sample(seed=next(seed), sample_shape=(N,))

    ref_mean = data.mean(axis=0)

    diff = data - ref_mean
    ref_cov = np.einsum('...d, ...e -> ...de', diff, diff)
    ref_cov = np.mean(ref_cov, axis=0)

    # Estimate parameters from data
    mle_distr = distr.compute_maximum_likelihood(data)
    assert np.allclose(mle_distr.mean(), ref_mean)
    assert np.allclose(mle_distr.covariance(), ref_cov, atol=1e-4)