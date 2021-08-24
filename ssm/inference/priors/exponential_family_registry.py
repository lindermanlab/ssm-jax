from collections import namedtuple

from .dirichlet import (_categorical_from_params, _categorical_suff_stats,
                        _dirichlet_from_stats,
                        _dirichlet_pseudo_obs_and_counts)
from .gamma import (_gamma_from_stats, _gamma_pseudo_obs_and_counts,
                    _poisson_from_params, _poisson_suff_stats)
from .normal_inverse_wishart import (_mvn_from_params, _mvn_suff_stats,
                                     _niw_from_stats,
                                     _niw_pseudo_obs_and_counts)

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
EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"] = ExponentialFamily(
    _niw_pseudo_obs_and_counts, _niw_from_stats, _mvn_from_params, _mvn_suff_stats
)

EXPFAM_DISTRIBUTIONS["IndependentPoisson"] = ExponentialFamily(
    _gamma_pseudo_obs_and_counts,
    _gamma_from_stats,
    _poisson_from_params,
    _poisson_suff_stats,
)

EXPFAM_DISTRIBUTIONS["Categorical"] = ExponentialFamily(
    _dirichlet_pseudo_obs_and_counts,
    _dirichlet_from_stats,
    _categorical_from_params,
    _categorical_suff_stats,
)
