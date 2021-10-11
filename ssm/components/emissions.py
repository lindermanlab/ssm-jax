"""
emissions.py
===============

PROTOTYPE:

General design notes:
=====================

- Emissions object should lighlty wrap TFP distributions to provide friendly interface upstream
    - interface should include:
        - .permute(permutation)                        ==> permute the latent states defined by the distribution
        - .exact_m_step(data, posterior, prior=None)   ==> (if possible) return an updated distribution
        - .sgd_update(data, posterior)                 ==> SGD on -log_prob objective (when m_step isn't available)
        - .initialize(data)                            ==> Initialize parameters of distribution in data-aware way
        - [...]
- Like the ssm.jax_refactor branch, we could bundle a list of different distributions / state

Issues:
=======
    - right now, there is redundancy for the ExpFam distribution M_steps.
        - can we quarter that off somewhere else?
    - is this abstraction layer necessary?
        - will try to clean up other stuff to see if this makes sense
    - how should approximate m steps (i.e. Laplace) be handled? 
        - can the details of the algorithm be abstracted away?
        - does it make sense for the inference details to be implemented here?
"""
import jax
import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import tensorflow_probability.substrates.jax as tfp

from jax.flatten_util import ravel_pytree

@register_pytree_node_class
class Emissions:
    def __init__(self, num_states, distribution):
        assert isinstance(distribution, tfp.distributions.Distribution)
        self.distribution = distribution
        self.num_states = num_states

    def exact_m_step(self, data, posterior, prior=None):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.distribution)
        aux_data = (self.num_states,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, = aux_data
        distribution = children
        return cls(num_states,
                   distribution=distribution)


@register_pytree_node_class
class GaussianEmissions(Emissions):
    def exact_m_step(self, data, posterior, prior=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        expfam = EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"]
        stats = tuple(
            np.einsum('tk,t...->k...', posterior.expected_states, s)
            for s in expfam.suff_stats(data))
        counts = np.sum(posterior.expected_states, axis=0)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())

@register_pytree_node_class
class PoissonEmissions(Emissions):
    def exact_m_step(self, data, posterior, prior=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        expfam = EXPFAM_DISTRIBUTIONS["IndependentPoisson"]
        stats = tuple(
            np.einsum('tk,t...->k...', posterior.expected_states, s)
            for s in expfam.suff_stats(data))
        counts = np.sum(posterior.expected_states, axis=0)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())

@register_pytree_node_class
class GaussianLinearRegressionEmissions(Emissions):
    def exact_m_step(self,data, posterior, prior=None):
        # Use exponential family stuff for the emissions
        expfam = EXPFAM_DISTRIBUTIONS["GaussianGLM"]

        # Extract expected sufficient statistics from posterior
        Ex = posterior.mean
        ExxT, _ = posterior.second_moments

        # Sum over time
        sum_x = Ex.sum(axis=0)
        sum_y = data.sum(axis=0)
        sum_xxT = ExxT.sum(axis=0)
        sum_yxT = data.T.dot(Ex)
        sum_yyT = data.T.dot(data)
        stats = (sum_x, sum_y, sum_xxT, sum_yxT, sum_yyT)
        counts = len(data)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())

@register_pytree_node_class
class GeneralizedLinearModelEmissions(Emissions):
    def exact_m_step(self, data, posterior, prior=None):
        assert "exact m step generally not possible for GLM models"
        return NotImplementedError
    
    def approx_m_step(self, data, posterior, rng):
        """Update the parameters of the emissions distribution via an approximate M step using samples from posterior.

        Uses BFGS to optimize the expected log probability of the emission via Monte Carlo estimate.

        For nonconjugate models like the GLM-LDS, we do not have a closed form expression for the objective nor solution
        to the M step parameter update for the emissions model. This is because the objective is technically an
        expectation under a Gaussian posterior on the latent states.

        We can approximate an update to our emissions distribution using a Monte Carlo estimate of this expectation,
        wherein we sample latent-state trajectories from our (potentially approximate) posterior and use these samples
        to compute the objective (the log probability of the data under the emissions distribution).

        Args:
            rng (jax.random.PRNGKey): JAX random seed.
            lds (LDS): The LDS model object.
            data (array, (num_timesteps, obs_dim)): Array of observed data.
            posterior (MultivariateNormalBlockTridiagonal):
                The LDS posterior object.
            prior (LDSPrior, optional): The prior distributions. Not yet supported. Defaults to None.

        Returns:
            emissions_distribution (tfp.distributions.Distribution):
                A new emissions distribution object with the updated parameters.
        """
        x_sample = posterior._sample(seed=rng)

        # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
        flat_emissions_distribution, unravel = ravel_pytree(self.distribution)
        def _objective(flat_emissions_distribution):
            # TODO: Consider proximal gradient descent to counter sampling noise
            emissions_distribution = unravel(flat_emissions_distribution)
            return -1 * np.mean(emissions_distribution.predict(x_sample).log_prob(data))

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            flat_emissions_distribution,
            method="BFGS")

        # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
        return unravel(optimize_results.x)


@register_pytree_node_class
class PoissonGLMEmissions(GeneralizedLinearModelEmissions):
    pass


@register_pytree_node_class
class BernoulliGLMEmissions(GeneralizedLinearModelEmissions):
    pass