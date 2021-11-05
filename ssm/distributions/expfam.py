from functools import partial

import jax.numpy as np
from jax import vmap
from jax.tree_util import tree_map

from ssm.distributions.mniw import MatrixNormalInverseWishart
from ssm.distributions.linreg import GaussianLinearRegression


__CONJUGATE_PRIORS = dict()

def register_prior(distribution, prior):
    __CONJUGATE_PRIORS[distribution] = prior


def get_prior(distribution):
    return __CONJUGATE_PRIORS[distribution]


class ConjugatePrior:
    r"""Interface for a conjugate prior distribution.

    TODO: give more detail
    """
    @classmethod
    def from_natural_parameters(cls, natural_params):
        """
        Construct an instance of the prior distribution given
        its natural parameters
        """
        raise NotImplementedError

    @property
    def natural_parameters(self):
        """Return the natural parameters of the distribution.
        These become pseudo-observations of the sufficient statistics
        of the conjugate distribution.
        """
        raise NotImplementedError


class ExponentialFamilyDistribution:
    r"""An interface for exponential family distributions
    with the necessary functionality for MAP estimation.

    ..math:
        p(x) = h(x) \exp\{t(x)^\top \eta - A(\eta)}

    where

    :math:`h(x)` is the base measure
    :math:`t(x)` are sufficient statistics
    :math:`\eta` are natural parameters
    :math:`A(\eta)` is the log normalizer

    """
    @classmethod
    def from_params(cls, params, **kwargs):
        """Create an instance parameters of the distribution
        with given parameters (e.g. the mode of a posterior distribution
        on those parameters). This function might have to do some conversion,
        e.g. from variances to scales.
        """
        raise NotImplementedError

    @staticmethod
    def sufficient_statistics(data, **kwargs):
        """
        Return the sufficient statistics for each datapoint in an array,
        This function should assume the leading dimensions give the batch
        size.
        """
        raise NotImplementedError

    @classmethod
    def compute_conditional_from_stats(cls, stats):
        return get_prior(cls).from_natural_parameters(stats)

    @classmethod
    def compute_conditional(cls, data, weights=None, prior=None):
        # Flatten the data and weights so we can vmap over them
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_data = flatten(data)

        # Collect sufficient statistics for each data point
        stats = vmap(cls.sufficient_statistics)(flat_data)

        # Sum the (weighted) sufficient statistics
        if weights is not None:
            flat_weights = flatten(weights)
            stats = tree_map(lambda x: np.einsum('nk,n...->k...', flat_weights, x), stats)
        else:
            stats = tree_map(partial(np.sum, axis=0), stats)

        # Add the natural parameters from the prior
        if prior is not None:
            stats = tree_map(np.add, stats, prior.natural_parameters)

        # Compute the conditional distribution given the stats
        return cls.compute_conditional_from_stats(stats)


### Matrix normal inverse Wishart / Linear Regression
def _mniw_from_stats(stats, counts):
    r"""Convert the statistics and counts back into MNIW parameters.

    stats = (Ex, Ey, ExxT, EyxT, EyyT)

    prior is on the concatenated weights and biases (W, b)

    Recall,
    .. math::
        n_1 = \nu_0 + n + p + 1
        s_1 = V_0^{-1}
        s_2 = M_0 V_0^{-1}
        s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top
    """
    Ex, Ey, ExxT, EyxT, EyyT = stats
    n = np.array(counts)

    # Pad the sufficient statistics to include the bias
    big_ExxT = np.concatenate([np.concatenate([ExxT,             Ex[..., :, None]],   axis=-1),
                               np.concatenate([Ex[..., None, :], n[..., None, None]], axis=-1)],
                              axis=-2)
    big_EyxT = np.concatenate([EyxT, Ey[..., None]], axis=-1)
    out_dim, in_dim = big_EyxT.shape[-2:]

    T = lambda X: np.swapaxes(X, -1, -2)
    nu0 = counts - out_dim - in_dim - 1
    V0 = np.linalg.inv(big_ExxT + 1e-4 * np.eye(in_dim))
    # M0 = big_EyxT @ V0
    M0 = T(np.linalg.solve(big_ExxT + 1e-4 * np.eye(in_dim), T(big_EyxT)))
    Psi0 = EyyT - M0 @ big_ExxT @ T(M0) + 1e-4 * np.eye(out_dim)

    # def _null_stats(operand):
    #     V0 = 1e16 * np.eye(in_dim)
    #     M0 = np.zeros_like(big_EyxT)
    #     Psi0 = np.eye(out_dim)
    #     return V0, M0, Psi0

    # def _stats(operand):
    #     # TODO: Use Cholesky factorization for these two steps
    #     V0 = np.linalg.inv(big_ExxT + 1e-16 * np.eye(in_dim))
    #     M0 = big_EyxT @ V0
    #     Psi0 = EyyT - M0 @ big_ExxT @ np.swapaxes(M0, -1, -2)
    #     return V0, M0, Psi0

    # V0, M0, Psi0 = lax.cond(np.allclose(big_ExxT, 0), _null_stats, _stats, operand=None)
    return MatrixNormalInverseWishart(M0, V0, nu0, Psi0)


def _mniw_pseudo_obs_and_counts(mniw):
    V0iM0T = np.linalg.solve(mniw.column_covariance, mniw.loc.T)
    stats = (np.linalg.inv(mniw.column_covariance),
             V0iM0T.T,
             mniw.scale + mniw.loc @ V0iM0T)
    counts = mniw.df + mniw.out_dim + mniw.in_dim + 1
    return stats, counts


def _gaussian_linreg_from_params(params):
    weights_and_bias, covariance_matrix = params
    weights, bias = weights_and_bias[:, :-1], weights_and_bias[:, -1]
    return GaussianLinearRegression(weights, bias, np.linalg.cholesky(covariance_matrix))


def _gaussian_linreg_suff_stats(data, covariates):
    return (covariates,
            data,
            np.einsum('...i,...j->...ij', covariates, covariates),
            np.einsum('...i,...j->...ij', data, covariates),
            np.einsum('...i,...j->...ij', data, data))
