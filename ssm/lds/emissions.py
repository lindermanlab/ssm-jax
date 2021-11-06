import jax
import jax.numpy as np
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.distributions.expfam import EXPFAM_DISTRIBUTIONS
from ssm.utils import sum_tuples

from ssm.distributions import GaussianLinearRegression, glm



@register_pytree_node_class
class Emissions:
    """
    Base class of emission distribution of an LDS

    .. math::
        p_t(y_t \mid x_t, u_t)

    where u_t are optional covariates.
    """
    def __init__(self,
                 weights=None,
                 bias=None,
                 scale_tril=None,
                 emissions_distribution: tfd.Distribution=None,
                 emissions_distribution_prior: tfd.Distribution=None) -> None:
        assert (weights is not None and \
                bias is not None and \
                scale_tril is not None) \
            or emissions_distribution is not None

        if weights is not None:
            self._distribution = GaussianLinearRegression(weights, bias, scale_tril)
        else:
            self._distribution = emissions_distribution

        if emissions_distribution_prior is None:
            pass  # TODO: implement default prior
        self._distribution_prior = emissions_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._distribution_prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def bias(self):
        return self._distribution.bias

    @property
    def scale_tril(self):
        return self._distribution.scale_tril

    def distribution(self, state, covariates=None):
        """
        Return the conditional distribution of emission y_t
        given state x_t and (optionally) covariates u_t.

        Note: covariates aren't supported yet.
        """
        if covariates is not None:
            # TODO: handle extra covariates
            raise NotImplementedError
        return self._distribution.predict(covariates=state)

    def m_step(self, dataset, posteriors, num_samples=1, rng=None):
        if rng is None:
            raise ValueError("PRNGKey needed for generic m-step")

        # Draw samples of the latent states
        sample_shape = () if num_samples == 1 else (num_samples,)
        x_sample = posteriors.sample(seed=rng, sample_shape=sample_shape)

        # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
        flat_emissions_distribution, unravel = ravel_pytree(self._distribution)
        def _objective(flat_emissions_distribution):
            # TODO: Consider proximal gradient descent to counter sampling noise
            emissions_distribution = unravel(flat_emissions_distribution)
            return -1 * np.mean(emissions_distribution.predict(x_sample).log_prob(dataset))

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            flat_emissions_distribution,
            method="BFGS"  # TODO: consider L-BFGS?
        )

        self._distribution = unravel(optimize_results.x)


@register_pytree_node_class
class GaussianEmissions(Emissions):
    def __init__(self,
                 weights=None,
                 bias=None,
                 scale_tril=None,
                 emissions_distribution: GaussianLinearRegression=None,
                 emissions_distribution_prior: tfd.Distribution=None) -> None:
        super(GaussianEmissions, self).__init__(
            weights, bias, scale_tril,
            emissions_distribution,
            emissions_distribution_prior
        )

    def m_step(self, dataset, posteriors, rng=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        # Use exponential family stuff for the emissions
        expfam = EXPFAM_DISTRIBUTIONS["GaussianLinearRegression"]

        def compute_stats_and_counts(data, posterior):
            # Extract expected sufficient statistics from posterior
            Ex = posterior.expected_states
            ExxT = posterior.expected_states_squared

            # Sum over time
            sum_x = Ex.sum(axis=0)
            sum_y = data.sum(axis=0)
            sum_xxT = ExxT.sum(axis=0)
            sum_yxT = data.T.dot(Ex)
            sum_yyT = data.T.dot(data)
            stats = (sum_x, sum_y, sum_xxT, sum_yxT, sum_yyT)
            counts = len(data)
            return stats, counts

        stats, counts = vmap(compute_stats_and_counts)(dataset, posteriors)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf
        counts = counts.sum(axis=0)

        if self._distribution_prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(self._distribution_prior.emissions_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        self._distribution = expfam.from_params(param_posterior.mode())


@register_pytree_node_class
class PoissonEmissions(Emissions):
    def __init__(self,
                 weights=None,
                 bias=None,
                 emissions_distribution: glm.PoissonGLM=None,
                 emissions_distribution_prior: tfd.Distribution=None) -> None:
        assert (weights is not None and \
                bias is not None) \
            or emissions_distribution is not None

        if weights is not None:
            self._distribution = glm.PoissonGLM(weights, bias)
        else:
            self._distribution = emissions_distribution

        if emissions_distribution_prior is None:
            pass  # TODO: implement default prior
        self._distribution_prior = emissions_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._distribution_prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def bias(self):
        return self._distribution.bias

    def distribution(self, state, covariates=None):
        """
        Return the conditional distribution of emission y_t
        given state x_t and (optionally) covariates u_t.

        Note: covariates aren't supported yet.
        """
        if covariates is not None:
            # TODO: handle extra covariates
            raise NotImplementedError
        return self._distribution.predict(covariates=state)
