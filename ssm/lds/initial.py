import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, tree_util, vmap
from jax.tree_util import register_pytree_node_class
from ssm.distributions.expfam import EXPFAM_DISTRIBUTIONS
from ssm.lds.components import ContinuousComponent
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

tfd = tfp.distributions

class InitialCondition:
    """
    Base class for initial state distributions of an LDS.

    ..math:
        p(x_1 \mid u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self):
        pass

    def distribution(self):
        """
        Return the distribution of z_1
        """
        raise NotImplementedError

    def log_probs(self, data):
        """
        Return [log Pr(z_1 = k) for k in range(num_states)]
        """
        return self.distribution().log_prob(np.arange(self.num_states))

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StandardInitialCondition(InitialCondition):
    """
    The standard model is a multivariate Normal distribution.
    (With covariance parameterized by the lower triagular scale
    cov = scale_tril @ scale_tril.T)
    """
    def __init__(self,
        initial_mean=None,
        initial_scale_tril=None,
        initial_distribution: tfd.MultivariateNormalTriL=None,
        initial_distribution_prior: tfd.Distribution=None) -> None:
        super(StandardInitialCondition, self).__init__()

        assert (initial_mean is not None and initial_scale_tril is not None) or initial_distribution is not None

        if initial_mean is not None:
            self._initial_distribution = tfd.MultivariateNormalTriL(loc=initial_mean, scale_tril=initial_scale_tril)
        else:
            self._initial_distribution = initial_distribution

        if initial_distribution_prior is None:
            pass  # TODO: implement default prior
        self._initial_distribution_prior = initial_distribution_prior

    def tree_flatten(self):
        children = (self._initial_distribution, self._initial_distribution_prior)
        aux_data = None
        return children, aux_data 

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(initial_distribution=distribution,
                   initial_distribution_prior=prior)
        
    @property
    def mean(self):
        return self._initial_distribution.loc

    def distribution(self):
       return self._initial_distribution

    def m_step(self, dataset, posteriors, prior=None):
        
        expfam = EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"]

        def compute_stats_and_counts(data, posterior):
            Ex = posterior.mean[0]
            ExxT = posterior.expected_states_squared[0]
            stats = (1.0, Ex, ExxT)
            counts = 1.0
            return stats, counts

        stats, counts = vmap(compute_stats_and_counts)(dataset, posteriors)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf
        counts = counts.sum(axis=0)

        if self._initial_distribution_prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(self._initial_distribution_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        self._initial_distribution = expfam.from_params(param_posterior.mode())
