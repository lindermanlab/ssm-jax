import jax
from jax._src.tree_util import tree_map
import jax.numpy as np
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.distributions import GaussianLinearRegression, glm


class Emissions:
    """
    Base class of emission distribution of an LDS

    .. math::
        p_t(y_t \mid x_t, u_t)

    where u_t are optional covariates.
    """
    @property
    def emissions_shape(self):
        raise NotImplementedError

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of emission y_t
        given state x_t and (optionally) covariates u_t.

        Args:
            state (float): continuous state
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        raise NotImplementedError

    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               num_samples=1,
               key=None):
        """Update the emissions distribution in-place using an M-step.

        Operates over a batch of data (posterior must have the same batch dim).

        Args:
            dataset (np.ndarray): the observed dataset
            posteriors (LDSPosterior): the HMM posteriors
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            num_samples (int): number of samples from posterior to use in a generic update
            key (jr.PRNGKey): random seed
        """
        # TODO: Implement generic m-step using samples of the posterior
        raise NotImplementedError


@register_pytree_node_class
class GaussianEmissions(Emissions):

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
        self._prior = emissions_distribution_prior

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

    @property
    def emissions_shape(self):
        return (self.weights.shape[-2],)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def bias(self):
        return self._distribution.bias

    @property
    def scale_tril(self):
        return self._distribution.scale_tril

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of emission y_t
        given state x_t and (optionally) covariates u_t.

        Args:
            state (float): continuous state
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        if covariates is not None:
            return self._distribution.predict(np.concatenate([state, covariates]))
        else:
            return self._distribution.predict(state)


    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               key=None):
        """Update the emissions distribution in-place using an exact M-step.

        Operates over a batch of data (posterior must have the same batch dim).

        Args:
            dataset (np.ndarray): the observed dataset
            posteriors (LDSPosterior): the HMM posteriors
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            num_samples (int): number of samples from posterior to use in a generic update
            key (jr.PRNGKey): random seed
        """
        def compute_stats_and_counts(data, posterior):
            # Extract expected sufficient statistics from posterior
            Ex = posterior.expected_states
            ExxT = posterior.expected_states_squared
            Ey = data
            EyxT = np.einsum('ti,tj->tij', data, Ex)
            EyyT = np.einsum('ti,tj->tij', data, data)

            # Concatenate with the covariates
            if covariates is not None:
                u = covariates
                Ex = np.column_stack((Ex, u))
                ExxT = vmap(lambda xi, xixiT, ui: \
                    np.block([[xixiT,            np.outer(xi, ui)],
                              [np.outer(ui, xi), np.outer(ui, ui)]]))(Ex, ExxT, u)
                EyxT = vmap(lambda yi, yixiT, ui: \
                    np.block([yixiT, np.outer(yi, ui)]))(Ey, EyxT, u)

            # Sum over time
            sum_x = Ex.sum(axis=0)
            sum_y = Ey.sum(axis=0)
            sum_xxT = ExxT.sum(axis=0)
            sum_yxT = EyxT.sum(axis=0)
            sum_yyT = EyyT.sum(axis=0)
            T = len(data)
            stats = (T, sum_xxT, sum_x, T, sum_yxT, sum_y, sum_yyT)
            return stats

        stats = vmap(compute_stats_and_counts)(data, posterior)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf

        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        conditional = ssmd.GaussianLinearRegression.compute_conditional_from_stats(stats)
        self._distribution = ssmd.GaussianLinearRegression.from_params(conditional.mode())


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

    @property
    def emissions_shape(self):
        return (self.weights.shape[-2],)

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

    def distribution(self, state, covariates=None, metadata=None):
        if covariates is not None:
            return self._distribution.predict(np.concatenate([state, covariates]))
        else:
            return self._distribution.predict(state)

    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               num_samples=1,
               key=None):
        if key is None:
            raise ValueError("PRNGKey needed for generic m-step")

        # Draw samples of the latent states
        state_samples = posterior.sample(seed=key, sample_shape=(num_samples,))

        # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
        flat_emissions_distribution, unravel = ravel_pytree(self._distribution)

        def _objective(flat_emissions_distribution):
            # TODO: Consider proximal gradient descent to counter sampling noise
            emissions_distribution = unravel(flat_emissions_distribution)

            def _lp_single(sample):
                if covariates is not None:
                    sample = np.concatenate([sample, covariates], axis=-1)
                return emissions_distribution.predict(sample).log_prob(data) / data.size

            return -1 * np.mean(vmap(_lp_single)(state_samples))

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            flat_emissions_distribution,
            method="BFGS"  # TODO: consider L-BFGS?
        )

        self._distribution = unravel(optimize_results.x)