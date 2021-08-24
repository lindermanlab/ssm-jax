import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
from collections import namedtuple


from ssm.distributions.niw import NormalInverseWishart

# Register the prior and likelihood functions
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


# Multivariate normal likelihood
def _mvn_from_params(params):
    loc, covariance = params
    scale_tril = np.linalg.cholesky(covariance)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)


def _mvn_suff_stats(data):
    return (np.ones(data.shape[:-1]), data, np.einsum("...i,...j->...ij", data, data))


EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"] = ExponentialFamily(
    _niw_pseudo_obs_and_counts, _niw_from_stats, _mvn_from_params, _mvn_suff_stats
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
