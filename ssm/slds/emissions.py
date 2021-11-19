import jax
from jax._src.tree_util import tree_map
import jax.numpy as np
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.distributions import GaussianLinearRegression, GaussianLinearRegressionPrior


@register_pytree_node_class
class Emissions:
    """
    Base class of emission distribution of an SLDS

    .. math::
        p_t(y_t \mid x_t, z_t, u_t)

    where u_t are optional covariates.
    """
    def __init__(self,
                 emissions_distribution: tfd.Distribution=None,
                 emissions_distribution_prior: tfd.Distribution=None) -> None:

        self._distribution = emissions_distribution
        self._prior = emissions_distribution_prior

    @property
    def emissions_shape(self):
        return self._distribution.event_shape

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)

    def m_step(self, dataset, posteriors, num_samples=1, rng=None):
        if rng is None:
            raise ValueError("PRNGKey needed for generic m-step")

        # Draw samples of the latent states
        state_samples = posteriors.sample(seed=rng, sample_shape=(num_samples,))

        # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
        flat_emissions_distribution, unravel = ravel_pytree(self._distribution)

        def _objective(flat_emissions_distribution):
            # TODO: Consider proximal gradient descent to counter sampling noise
            emissions_distribution = unravel(flat_emissions_distribution)

            # Compute log probability for a single sample
            def _lp_single(sample):
                z = sample["discrete_state"]
                x = sample["continuous_state"]
                conditional = emissions_distribution[z].predict(covariates=x)
                return conditional.log_prob(dataset) / dataset.size

            # Objective is the negative mean over samples
            return -1 * np.mean(vmap(_lp_single)(state_samples))

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
                 emissions_distribution_prior: GaussianLinearRegressionPrior=None) -> None:
        if weights is not None:
            emissions_distribution = GaussianLinearRegression(weights, bias, scale_tril)

        if emissions_distribution_prior is None:
            pass  # TODO: implement default prior

        super().__init__(emissions_distribution,
                         emissions_distribution_prior)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def bias(self):
        return self._distribution.bias

    @property
    def scale(self):
        return self._distribution.scale

    def distribution(self, state, covariates=None, metadat=None):
        z = state["discrete"]
        x = state["continuous"]
        return self._distribution[z].predict(x)

    def m_step(self, dataset, posteriors, rng=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        def compute_stats_and_counts(data, posterior):
            # Extract expected sufficient statistics from posterior
            Ez = posterior.discrete_posterior.expected_states
            Ex = posterior.continuous_posterior.expected_states
            ExxT = posterior.continuous_posterior.expected_states_squared

            # Sum over time
            sum_x = np.einsum('tk,ti->ki', Ez, Ex)
            sum_y = np.einsum('tk,ti->ki', Ez, data)
            sum_xxT = np.einsum('tk,tij->kij', Ez, ExxT)
            sum_yxT = np.einsum('tk,ti,tj->kij', Ez, data, Ex)
            sum_yyT = np.einsum('tk,ti,tj->kij', Ez, data, data)
            T = np.sum(Ez, axis=0)
            stats = (T, sum_xxT, sum_x, T, sum_yxT, sum_y, sum_yyT)
            return stats

        stats = vmap(compute_stats_and_counts)(dataset, posteriors)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf

        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        conditional = ssmd.GaussianLinearRegression.compute_conditional_from_stats(stats)
        self._distribution = ssmd.GaussianLinearRegression.from_params(conditional.mode())
