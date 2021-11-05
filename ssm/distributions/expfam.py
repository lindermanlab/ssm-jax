import jax.numpy as np
from jax import lax, vmap
from jax.tree_util import tree_map

from tensorflow_probability.substrates import jax as tfp
from collections import namedtuple

from ssm.distributions.niw import NormalInverseWishart
from ssm.distributions.mniw import MatrixNormalInverseWishart
from ssm.distributions.linreg import GaussianLinearRegression
from ssm.utils import sum_tuples

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


EXPFAM_DISTRIBUTIONS = dict()


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
    def log_normalizer(params, **kwargs):
        """
        Return the log normalizer of the distribution.
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
    def fit_with_stats(cls,
                       sufficient_statistics,
                       num_datapoints,
                       prior=None,
                       **kwargs):
        """Compute the maximum a posteriori (MAP) estimate of the distribution
        parameters, given the sufficient statistics of the data and the number
        of datapoints.
        """
        # Compute the posterior distribution given sufficient statistics
        posterior_stats = sufficient_statistics
        posterior_counts = num_datapoints

        # Add prior stats if given
        if prior is not None:
            posterior_stats = sum_tuples(prior.pseudo_obs, posterior_stats)
            posterior_counts += prior.pseudo_counts

        # Construct the posterior
        posterior_class = get_expfam(cls)
        posterior = posterior_class.from_stats(posterior_stats, posterior_counts, **kwargs)

        # Return an instance of this distribution using the posterior mode parameters
        return cls.from_params(posterior.mode(), **kwargs)

    @classmethod
    def fit(cls, dataset, weights=None, prior=None, **kwargs):
        """Compute the maximum a posteriori (MAP) estimate of the distribution
        parameters.  For uninformative priors, this reduces to the maximum
        likelihood estimate.
        """
        # Compute the sufficient statistics and the number of datapoints
        suff_stats = None
        num_datapoints = 0
        for data_dict, these_weights in zip(dataset, weights):
            these_stats = cls.sufficient_statistics(**data_dict, **kwargs)

            # weight the statistics if weights are given
            if these_weights is not None:
                these_stats = tuple(np.tensordot(these_weights, s, axes=(0, 0))
                                    for s in these_stats)
            else:
                these_stats = tuple(s.sum(axis=0) for s in these_stats)

            # add to our accumulated statistics
            suff_stats = sum_tuples(suff_stats, these_stats)

            # update the number of datapoints
            num_datapoints += these_weights.sum()

        return cls.fit_with_stats(suff_stats, num_datapoints, prior=prior, **kwargs)


def compute_conditional_with_stats(distribution_name,
                                   sufficient_statistics,
                                   num_datapoints,
                                   prior=None,
                                   **kwargs):
    """Compute the maximum a posteriori (MAP) estimate of the distribution
    parameters, given the sufficient statistics of the data and the number
    of datapoints.
    """
    expfam = EXPFAM_DISTRIBUTIONS[distribution_name]

    # Compute the posterior distribution given sufficient statistics
    stats = sufficient_statistics
    counts = num_datapoints

    # Add prior stats if given
    if prior is not None:
        prior_stats, prior_counts = expfam.prior_pseudo_obs_and_counts(prior)
        stats = sum_tuples(prior_stats, stats)
        counts += prior_counts

    # Construct the conditional distribution
    return expfam.posterior_from_stats(stats, counts)


def compute_conditional(distribution_name,
                        data,
                        weights=None,
                        prior=None,
                        **kwargs):
    """Compute the conditional distribution over parameters of a distribution
    given data.

    distribution_name: string key into `EXPFAM_DISTRIBUTIONS`
    data: (..., D) array of data assumed iid from distribution
    weights: (...,) array of weights for each data point
    prior: conjugate prior object
    """
    expfam = EXPFAM_DISTRIBUTIONS[distribution_name]

    # Compute the sufficient statistics and the number of datapoints
    stats = vmap(expfam.suff_stats)(data.reshape(-1, data.shape[-1]))


    # weight the statistics if weights are given
    if weights is not None:
        stats = tree_map(lambda x: np.einsum('n,n...->...', weights.ravel(), x), stats)
        counts = np.sum(weights)
    else:
        stats = tree_map(lambda x: np.sum(x, axis=0), stats)
        counts = len(data.reshape(-1, data.shape[-1]))

    return compute_conditional_with_stats(distribution_name,
                                          stats,
                                          counts,
                                          prior=prior,
                                          **kwargs)


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


EXPFAM_DISTRIBUTIONS["MultivariateNormal"] = ExponentialFamily(
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
    (shape,) = stats
    return tfp.distributions.Gamma(shape, rate=counts)


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
