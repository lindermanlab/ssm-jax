import jax.numpy as np
from jax import lax
from tensorflow_probability.substrates import jax as tfp
from collections import namedtuple

from ssm.distributions.niw import NormalInverseWishart
from ssm.distributions.mniw import MatrixNormalInverseWishart
from ssm.distributions.linreg import GaussianLinearRegression


# Register the prior and likelihood functions
# TODO: bundle these into a class
ExponentialFamily = namedtuple(
    "ExponentialFamily",
    [
        "prior_pseudo_obs_and_counts",
        "posterior_from_stats",
        "from_params",
        "suff_stats",
    ],
)

# class ExponentialFamily:
#     def __init__(self):
#         pass

#     def posterior_from_stats(self, stats, counts):
#         return NotImplementedError

#     def prior_pseudo_obs_and_counts(self):
#         return NotImplementedError

#     def from_params(self):
#         return NotImplementedError

#     def suff_stats(self):
#         return NotImplementedError


EXPFAM_DISTRIBUTIONS = dict()


### Normal inverse Wishart / Multivariate Normal
def _niw_from_stats(stats, counts):
    s_1, s_2, s_3 = stats
    dim = s_2.shape[-1]

    mean_precision = s_1
    # loc = lax.cond(mean_precision > 0,
    #             lambda x: s_2 / mean_precision,
    #             lambda x: np.zeros_like(s_2),
    #             None)
    loc = np.einsum("...i,...->...i", s_2, 1 / mean_precision)
    scale = s_3 - np.einsum("...,...i,...j->...ij", mean_precision, loc, loc)
    df = counts - dim - 2
    return NormalInverseWishart(loc, mean_precision, df, scale)


def _niw_pseudo_obs_and_counts(niw):
    pseudo_obs = (
        niw.mean_precision,
        niw.mean_precision * niw.loc,
        niw.scale
        + niw.mean_precision * np.einsum("...i,...j->...ij", niw.loc, niw.loc),
    )

    pseudo_counts = niw.df + niw.dim + 1
    return pseudo_obs, pseudo_counts


def _mvn_from_params(params):
    loc, covariance = params
    scale_tril = np.linalg.cholesky(covariance)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)


def _mvn_suff_stats(data):
    return (np.ones(data.shape[:-1]), data, np.einsum("...i,...j->...ij", data, data))


EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"] = ExponentialFamily(
    _niw_pseudo_obs_and_counts, _niw_from_stats, _mvn_from_params, _mvn_suff_stats
)


### Matrix normal inverse Wishart / Linear Regression
def _mniw_from_stats(stats, counts):
    r"""Convert the statistics and counts back into MNIW parameters.

    stats = (Ex, Ey, ExxT, EyxT, EyyT)

    prior is on the concatenated weights and biases (W, b)

    Recall,
    ..math::
        n_1 = \nu_0 + n + p + 1
        s_1 = V_0^{-1}
        s_2 = M_0 V_0^{-1}
        s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top
    """
    Ex, Ey, ExxT, EyxT, EyyT = stats
    T = counts

    # Pad the sufficient statistics to include the bias
    big_ExxT = np.row_stack([np.column_stack([ExxT, Ex]),
                             np.concatenate( [Ex.T, np.array([T])])])
    big_EyxT = np.column_stack([EyxT, Ey])
    out_dim, in_dim = big_EyxT.shape[-2:]

    nu0 = counts - out_dim - in_dim - 1
    def _null_stats(operand):
        V0 = 1e16 * np.eye(in_dim)
        M0 = np.zeros_like(big_EyxT)
        Psi0 = np.eye(out_dim)
        return V0, M0, Psi0

    def _stats(operand):
        # TODO: Use Cholesky factorization for these two steps
        V0 = np.linalg.inv(big_ExxT + 1e-16 * np.eye(in_dim))
        M0 = big_EyxT @ V0
        Psi0 = EyyT - M0 @ big_ExxT @ M0.T
        return V0, M0, Psi0

    V0, M0, Psi0 = lax.cond(np.allclose(big_ExxT, 0), _null_stats, _stats, operand=None)
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


EXPFAM_DISTRIBUTIONS["GaussianLinearRegression"] = ExponentialFamily(
    _mniw_pseudo_obs_and_counts,
    _mniw_from_stats,
    _gaussian_linreg_from_params,
    _gaussian_linreg_suff_stats,
)

EXPFAM_DISTRIBUTIONS["GaussianGLM"] = ExponentialFamily(
    _mniw_pseudo_obs_and_counts,
    _mniw_from_stats,
    _gaussian_linreg_from_params,
    _gaussian_linreg_suff_stats,
)


### Gamma / Poisson
def _gamma_from_stats(stats, counts):
    (alpha,) = stats
    return tfp.distributions.Gamma(alpha, rate=counts[:, None])


def _gamma_pseudo_obs_and_counts(gamma):
    return (gamma.concentration,), gamma.rate


def _poisson_from_params(params):
    return tfp.distributions.Independent(
        tfp.distributions.Poisson(rate=params), reinterpreted_batch_ndims=1
    )


def _poisson_suff_stats(data):
    return (data,)


EXPFAM_DISTRIBUTIONS["IndependentPoisson"] = ExponentialFamily(
    _gamma_pseudo_obs_and_counts,
    _gamma_from_stats,
    _poisson_from_params,
    _poisson_suff_stats,
)

### Dirichlet / Categorical
def _dirichlet_from_stats(stats, counts):
    concentration = stats[0]
    return tfp.distributions.Dirichlet(concentration)


def _dirichlet_pseudo_obs_and_counts(dirichlet):
    return (dirichlet.concentration,), 0


def _categorical_from_params(params):
    return tfp.distributions.Categorical(probs=params)


def _categorical_suff_stats(data):
    num_classes = int(data.max()) + 1
    return (data[..., None] == np.arange(num_classes),)


EXPFAM_DISTRIBUTIONS["Categorical"] = ExponentialFamily(
    _dirichlet_pseudo_obs_and_counts,
    _dirichlet_from_stats,
    _categorical_from_params,
    _categorical_suff_stats,
)
