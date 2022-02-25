from __future__ import annotations
from re import S
import jax.numpy as np
from jax import vmap
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from contextlib import contextmanager
import jax.scipy.optimize
import copy
import ssm

from ssm.module import Module

from tensorflow_probability.substrates import jax as tfp
from flax.core.frozen_dict import freeze, FrozenDict
import ssm.distributions as ssmd
tfd = tfp.distributions

from ssm.utils import tfp_dist_to_unconst_params, unconst_params_to_tfp_dist


class Emissions(Module):
    """
    Base class of emission distribution of an HMM

    ..  math::
        p_t(x_t \mid z_t, u_t)

    where u_t are optional covariates.
    """
    def __init__(self, num_states: int) -> None:
        self.num_states = num_states

    @property
    def emissions_shape(self):
        raise NotImplementedError

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of emission x_t
        given state z_t and (optionally) covariates u_t.
        """
        raise NotImplementedError

    def log_likelihoods(self, data, covariates=None, metadata=None):
        """
        Compute log p(x_t | z_t=k) for all t and k.
        """
        inds = np.arange(self.num_states)
        return vmap(lambda k: self.distribution(k, covariates=covariates, metadata=metadata).log_prob(data))(inds).T

    # TODO: ensure_has_batched_dim?
    def m_step(self, data, posterior, covariates=None, metadata=None) -> Emissions:
        """By default, try to optimize the emission distribution via generic
        gradient-based optimization of the expected log likelihood.

        This function assumes that the Emissions subclass is a PyTree and
        that all of its leaf nodes are unconstrained parameters.

        Args:
            data (np.ndarray): the observed data
            posterior (HMMPosterior): the HMM posterior
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            emissions (ExponentialFamilyEmissions): updated emissions object
        """
        
        parameters = self._parameters
        flat_parameters, unravel = ravel_pytree(parameters)
        
        def _objective(flat_parameters):

            parameters = unravel(flat_parameters)

            with self.inject(parameters):
                f = lambda data, expected_states: \
                    np.sum(self.log_likelihoods(data, covariates=covariates, metadata=metadata) * expected_states)
                lp = vmap(f)(data, posterior.expected_states).sum()
                
            return -lp / data.size

        results = jax.scipy.optimize.minimize(
            _objective,
            flat_parameters,
            method="bfgs",
            options=dict(maxiter=100))
        
        self._parameters = unravel(results.x)


class ExponentialFamilyEmissions(Emissions):
    _emissions_distribution_class = None

    def __init__(self,
                 num_states: int,
                 emissions_distribution: ssmd.ExponentialFamilyDistribution=None,
                 emissions_distribution_prior: ssmd.ConjugatePrior=None) -> None:
        """Exponential Family Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            means (np.ndarray, optional): state-dependent emission means. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
            emissions_distribution (ssmd.ExponentialFamilyDistribution, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.ConjugatePrior, optional): initialized emissions distribution prior.
                Defaults to None.
        """

        super(ExponentialFamilyEmissions, self).__init__(num_states)
        self._distribution = emissions_distribution
        self._prior = emissions_distribution_prior

    @property
    def _parameters(self):
        return freeze(tfp_dist_to_unconst_params(self._distribution))
        
    @_parameters.setter
    def _parameters(self, params):
        self._distribution = unconst_params_to_tfp_dist(self._emissions_distribution_class,
                                                        params)
        
    @property
    def _hyperparameters(self):
        return freeze(dict(prior=self._prior,
                           num_states=self.num_states))
    
    @_hyperparameters.setter
    def _hyperparameters(self, hyperparams):
        self._prior = hyperparams["prior"]
        self.num_states = hyperparams["num_states"]
        
    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)

    @property
    def emissions_shape(self):
        return self._distribution.event_shape

    def distribution(self, state: int, covariates=None, metadata=None) -> ssmd.MultivariateNormalTriL:
        """Get the emissions distribution at the provided state.

        Args:
            state (int): discrete state
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        return self._distribution[state]

    def m_step(self, dataset, posteriors, covariates=None, metadata=None) -> ExponentialFamilyEmissions:
        """Update the emissions distribution using an M-step.

        Operates over a batch of data (posterior must have the same batch dim).

        Args:
            dataset (np.ndarray): the observed dataset
            posteriors (HMMPosterior): the HMM posteriors
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            emissions (ExponentialFamilyEmissions): updated emissions object
        """
        conditional = self._emissions_distribution_class.compute_conditional(
            dataset, weights=posteriors.expected_states, prior=self._prior)
        self._distribution = self._emissions_distribution_class.from_params(
            conditional.mode())


@register_pytree_node_class
class BernoulliEmissions(ExponentialFamilyEmissions):
    _emissions_distribution_class = ssmd.IndependentBernoulli

    def __init__(self,
                 num_states: int,
                 probs: np.ndarray=None,
                 emissions_distribution: ssmd.MultivariateNormalTriL=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        """Gaussian Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            probs (np.ndarray, optional): state-dependent emission probabilities. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """

        assert probs is not None or emissions_distribution is not None

        if probs is not None:
            emissions_distribution = ssmd.IndependentBernoulli(probs=probs)

        if emissions_distribution_prior is None:
            emissions_distribution_prior = ssmd.Beta(1.1, 1.1)

        super(BernoulliEmissions, self).__init__(num_states,
                                                 emissions_distribution,
                                                 emissions_distribution_prior)


@register_pytree_node_class
class GaussianEmissions(ExponentialFamilyEmissions):
    _emissions_distribution_class = ssmd.MultivariateNormalFullCovariance

    def __init__(self,
                 num_states: int,
                 means: np.ndarray=None,
                 covariances: np.ndarray=None,
                 prior_loc=None,
                 prior_mean_precision=None,
                 prior_df=None,
                 prior_scale=None,
                 emissions_distribution: ssmd.MultivariateNormalFullCovariance=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        """Gaussian Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            means (np.ndarray, optional): state-dependent emission means. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
        """
        
        if emissions_distribution is not None:
            emissions_distribution = emissions_distribution
        else:
            assert (means is not None and covariances is not None)
            emissions_distribution = ssmd.MultivariateNormalFullCovariance(means, covariances)
            
        if emissions_distribution_prior is not None:
            emissions_distribution_prior = emissions_distribution_prior
            
        else:
            
            # by default, construct a uninformative prior
            if (None in [prior_loc, prior_mean_precision, prior_df, prior_scale]):
                
                emissions_dim = means.shape[-1]
                
                prior_loc = np.zeros((emissions_dim,))
                prior_mean_precision = 0.
                prior_df = -emissions_dim - 2
                prior_scale = np.zeros((emissions_dim,emissions_dim))
                
            emissions_distribution_prior = ssmd.NormalInverseWishart(loc=prior_loc,
                                                    mean_precision=prior_mean_precision,
                                                    df=prior_df,
                                                    scale=prior_scale)
        
        super(GaussianEmissions, self).__init__(num_states,
                                                 emissions_distribution,
                                                 emissions_distribution_prior)

@register_pytree_node_class
class PoissonEmissions(ExponentialFamilyEmissions):
    _emissions_distribution_class = ssmd.IndependentPoisson

    def __init__(self,
                 num_states: int,
                 rates: np.ndarray=None,
                 emissions_distribution: ssmd.IndependentPoisson=None,
                 emissions_distribution_prior: ssmd.Gamma=None) -> None:
        """Poisson Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            rates (np.ndarray, optional): state-dependent Poisson rates. Defaults to None.
            emissions_distribution (tfd.Distribution, optional): pre-initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (tfd.Gamma, optional): pre-initialized emissions distribution prior.
                Defaults to None.
        """
        assert rates is not None or emissions_distribution is not None
        if rates is not None:
            emissions_distribution = ssmd.IndependentPoisson(rates)

        super(PoissonEmissions, self).__init__(num_states,
                                               emissions_distribution,
                                               emissions_distribution_prior)
