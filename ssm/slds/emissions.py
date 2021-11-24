import jax
from jax._src.tree_util import tree_map
import jax.numpy as np
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.distributions import GaussianLinearRegression, GaussianLinearRegressionPrior, PoissonGLM


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

    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               key=None):
        raise NotImplementedError


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
    def biases(self):
        return self._distribution.bias

    @property
    def covariances(self):
        return self._distribution.covariance

    def distribution(self, state, covariates=None, metadat=None):
        z = state["discrete"]
        x = state["continuous"]
        return self._distribution[z].predict(x)

    def m_step(self, data, posterior, covariates=None, metadata=None, key=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        # Extract expected sufficient statistics from posterior
        Ez = posterior.discrete_posterior.expected_states
        Ex = posterior.continuous_posterior.expected_states
        ExxT = posterior.continuous_posterior.expected_states_squared

        # Sum over time
        sum_x = np.einsum('btk,bti->ki', Ez, Ex)
        sum_y = np.einsum('btk,bti->ki', Ez, data)
        sum_xxT = np.einsum('btk,btij->kij', Ez, ExxT)
        sum_yxT = np.einsum('btk,bti,btj->kij', Ez, data, Ex)
        sum_yyT = np.einsum('btk,bti,btj->kij', Ez, data, data)
        T = np.einsum('btk->k', Ez)
        stats = (T, sum_xxT, sum_x, T, sum_yxT, sum_y, sum_yyT)

        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        conditional = ssmd.GaussianLinearRegression.compute_conditional_from_stats(stats)
        self._distribution = ssmd.GaussianLinearRegression.from_params(conditional.mode())


@register_pytree_node_class
class PoissonEmissions(Emissions):
    def __init__(self,
                 weights=None,
                 bias=None,
                 emissions_distribution: PoissonGLM=None,
                 emissions_distribution_prior: tfd.Distribution=None) -> None:
        if weights is not None:
            emissions_distribution = PoissonGLM(weights, bias)

        if emissions_distribution_prior is None:
            pass  # TODO: implement default prior

        super().__init__(emissions_distribution,
                         emissions_distribution_prior)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def biases(self):
        return self._distribution.bias

    def distribution(self, state, covariates=None, metadat=None):
        z = state["discrete"]
        x = state["continuous"]
        if covariates is not None:
            x = np.concatenate([x, covariates])
        return self._distribution[z].predict(x)

    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               num_samples=1,
               key=None,
               num_iters=50):
        assert key is not None, "PRNGKey needed for generic m-step"

        # Draw samples of the latent states
        state_samples = posterior.sample(seed=key, sample_shape=(num_samples,))

        # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
        flat_emissions_distribution, unravel = ravel_pytree(self._distribution)

        def _objective(flat_emissions_distribution):
            # TODO: Consider proximal gradient descent to counter sampling noise
            emissions_distribution = unravel(flat_emissions_distribution)

            # Compute log probability for a single sample
            def _lp_single(sample):
                z = sample["discrete"]
                x = sample["continuous"]
                if covariates is not None:
                    x = np.concatenate([x, covariates], axis=-1)
                conditional = emissions_distribution[z].predict(x)
                return conditional.log_prob(data) / data.size

            # Objective is the negative mean over samples
            return -1 * np.mean(vmap(_lp_single)(state_samples))

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            flat_emissions_distribution,
            method="l-bfgs-experimental-do-not-rely-on-this",
            options=dict(maxiter=num_iters))

        self._distribution = unravel(optimize_results.x)
