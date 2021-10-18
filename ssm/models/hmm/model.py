"""
HMM Model Classes
=================

Module defining model behavior for Hidden Markov Models (HMMs).
"""
from typing import Any

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
from jax import jit, value_and_grad, vmap
from jax.tree_util import register_pytree_node_class, tree_map
from ssm.models.base import SSM
from ssm.utils import Verbosity
from tensorflow_probability.substrates import jax as tfp

Array = Any

from ssm.inference.em import em
from ssm.inference.message_passing import (hmm_expected_states,
                                           hmm_log_normalizer)

from .dynamics import CategoricalDynamics
from .emissions import GaussianEmissions, PoissonEmissions
from .initial_distributions import CategoricalInitialDistribution
from .posterior import HMMPosterior


@register_pytree_node_class
class HMM(SSM):
    
    def __init__(self, num_states: int,
                 initial_distribution: tfp.distributions.Categorical,
                 transition_distribution: tfp.distributions.Categorical,
                 emissions_distribution: tfp.distributions.Distribution):
        """Class for Hidden Markov Model (HMM).

        Args:
            num_states (int): Number of discrete latent states.
            initial_distribution (tfp.distributions.Categorical): The distribution over the initial state.
            transition_distribution (tfp.distributions.Categorical): The transition distribution.
            emissions_distribution (tfp.distributions.Distribution): The emissions distribution.
        """
        self.num_states = num_states

        # Initalize initial_distribution, dynamics, and emissions components
        # TODO: this should be a lookup table based on constructor params
        self.initials = CategoricalInitialDistribution(initial_distribution, num_states)
        self.dynamics = CategoricalDynamics(transition_distribution, num_states)        
        if isinstance(emissions_distribution, tfp.distributions.Distribution):
            if emissions_distribution.name == "IndependentPoisson":
                self.emissions = PoissonEmissions(emissions_distribution, num_states)
            elif emissions_distribution.name == "MultivariateNormalTriL":
                self.emissions = GaussianEmissions(emissions_distribution, num_states)

    def initial_distribution(self):
        return self.initials.distribution

    def dynamics_distribution(self, state):
        return self.dynamics.distribution[state]

    def emissions_distribution(self, state):
        return self.emissions.distribution[state]

    @property
    def initial_state_probs(self):
        return self.initials.distribution.probs_parameter()

    @property
    def transition_matrix(self):
        return self.dynamics.distribution.probs_parameter()

    def natural_parameters(self, data: Array):
        """Obtain the natural parameters for the HMM given observation data.
        
        The natural parameters for an HMM are:
            - log probability of the initial state distribution
            - log probablity of the transitions (log transition matrix)
            - log likelihoods of the emissions data

        Args:
            data (Array): Observed data array: ``(time, obs_dim)``.

        Returns:
            log_initial_state_distn (Array): log probability of the initial state distribution
            log_transition_matrix (Array): log of transition matrix
            log_likelihoods (Array): log probability of emissions
        """
        log_initial_state_distn = self.initials.distribution.logits_parameter()
        log_transition_matrix = self.dynamics.distribution.logits_parameter()
        log_transition_matrix -= spsp.logsumexp(log_transition_matrix, axis=1, keepdims=True)
        log_likelihoods = vmap(lambda k:
                               vmap(lambda x: self.emissions.distribution[k].log_prob(x))(data)
                               )(np.arange(self.num_states)).T

        return log_initial_state_distn, log_transition_matrix, log_likelihoods

    ### Methods for inference

    def posterior(self, data):
        marginal_likelihood, (Ez0, Ezzp1, Ez) = hmm_expected_states(*self.natural_parameters(data))
        return HMMPosterior(marginal_likelihood, Ez, Ezzp1)

    def e_step(self, data):
        return self.posterior(data)

    def fit_using_posterior(self, data, posterior, prior=None):
        new_initial_distribution = self.initials.exact_m_step(data, posterior, prior=prior)
        new_dynamics_distribution = self.dynamics.exact_m_step(data, posterior, prior=prior)
        new_emissions_distribution = self.emissions.exact_m_step(data, posterior, prior=prior)

        # TODO: should this return a new object in this class-based formulation?
        return HMM(self.num_states,
                new_initial_distribution,
                new_dynamics_distribution,
                new_emissions_distribution)

    def m_step(self, data, posterior, prior=None):
        return self.fit_using_posterior(data, posterior, prior=None)

    def marginal_likelihood(self, data, posterior=None):
        if posterior is None:
            posterior = self.posterior(data)
        return posterior.marginal_likelihood

    def expected_log_joint(self, data, posterior):
        """The expected log joint probability of an HMM given a posterior over the latent states.
            
            .. math::
                \mathbb{E}_{q(z)} \left[\log p(x, z, \\theta)\\right]
                
            where,
            
            .. math:: 
                q(z) = p(z|x, \\theta).
                
            Recall that this is the component of the ELBO that is dependent upon the parameters.
            
            .. math::
                \mathcal{L}(q, \\theta) = \mathbb{E}_{q(z)} \left[\log p(x, z, \\theta) - \log q(z) \\right]

            Args:
                data ([type]): [description]
                posterior ([type]): [description]

            Returns:
                elp ([type]): expected log probability
            """
        log_initial_state_probs, log_transition_matrix, log_likelihoods = self.natural_parameters(data)
        elp = np.sum(posterior.expected_states[0] * log_initial_state_probs)
        elp += np.sum(posterior.expected_transitions * log_transition_matrix)
        elp += np.sum(posterior.expected_states * log_likelihoods)
        return elp

    def fit(self, data, method="em", num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG):
        model = self
        single_batch_mode = False
        kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

        # ensure data has a batch dimension
        if data.ndim == 2:
            single_batch_mode = True
            data = np.expand_dims(data, axis=0)
        assert data.ndim == 3, "data must have a batch dimension (B, T, N)"

        if method == "em":
            log_probs, model, posteriors = em(model, data, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        # squeeze first dimension
        if single_batch_mode:
            posteriors = tree_map(lambda x: x[0], posteriors)

        return log_probs, model, posteriors

    def tree_flatten(self):
            children = (self.initials.distribution,
                        self.dynamics.distribution,
                        self.emissions.distribution)
            aux_data = (self.num_states,)
            return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        initial_distribution, dynamics_distribution, emissions_distribution = children
        num_states, = aux_data
        return cls(num_states,
                   initial_distribution,
                   dynamics_distribution,
                   emissions_distribution)

class HMMConjugatePrior(object):
    """ TODO @schlagercollin
    """
    def log_prob(self, hmm):
        raise NotImplementedError

    @property
    def initial_prior(self):
        return self._initial_prior

    @property
    def transition_prior(self):
        return self._transition_prior

    @property
    def emissions_prior(self):
        return self._emissions_prior


# TODO: can we clean up these functions so that they are more usable?

# Helper functions to construct HMMs
def _make_standard_hmm(num_states, initial_state_probs=None,
                       initial_state_logits=None,
                       transition_matrix=None,
                       transition_logits=None):
    # Set up the initial state distribution and prior
    if initial_state_logits is None:
        if initial_state_probs is None:
            initial_state_logits = np.zeros(num_states)
        else:
            initial_state_logits = np.log(initial_state_probs)

    initial_dist = tfp.distributions.Categorical(logits=initial_state_logits)

    # Set up the transition matrix and prior
    if transition_logits is None:
        if transition_matrix is None:
            transition_logits = np.zeros((num_states, num_states))
        else:
            transition_logits = np.log(transition_matrix)

    transition_dist = tfp.distributions.Categorical(logits=transition_logits)

    return initial_dist, transition_dist


# Gaussian HMM
def make_gaussian_hmm(num_states: int,
                      emission_dim: int,
                      initial_state_probs: Array=None,
                      initial_state_logits: Array=None,
                      transition_matrix: Array=None,
                      transition_logits: Array=None,
                      emission_means: Array=None,
                      emission_covariances: Array=None,
                      emission_scale_trils: Array=None):
    """Helper function to create a Gaussian HMM given distribution parameters.

    Args:
        num_states (int): The number of discrete states in the HMM.
        emission_dim (int): The dimension of the output emissions.
        initial_state_probs (Array, optional): A ``(num_states,)`` array specifying the probabilities 
            of the initial state. Defaults to None.
        initial_state_logits (Array, optional): A ``(num_states,)`` array specifying the logits
            of the initial state. Defaults to None.
        transition_matrix (Array, optional): A ``(num_states, num_states)`` array specifying the transition
            probabilities. Defaults to None.
        transition_logits (Array, optional): A ``(num_states, num_states)`` array specifying the transition
            logits. Defaults to None.
        emission_means (Array, optional): A ``(num_states, obs_dim)`` array specifying the means of the 
            emissions distributions. Defaults to None.
        emission_covariances (Array, optional): A ``(num_states, obs_dim, obs_dim)`` array specifying the
            covariances of the emissions distributions. Defaults to None.
        emission_scale_trils (Array, optional): Lower-triagonal tensor specifying the scale of the emissions
            distribution. Defaults to None. 

    Returns:
        gaussian_hmm [HMM]: An intialized Gaussian HMM object. 
    """
    # Initialize the basics
    initial_dist, transition_dist = \
        _make_standard_hmm(num_states,
                           initial_state_probs=initial_state_probs,
                           initial_state_logits=initial_state_logits,
                           transition_matrix=transition_matrix,
                           transition_logits=transition_logits)

    # Initialize the Gaussian emissions
    if emission_means is None:
        emission_means = np.zeros((num_states, emission_dim))

    if emission_scale_trils is None:
        if emission_covariances is None:
            emission_scale_trils = np.tile(np.eye(emission_dim), (num_states, 1, 1))
        else:
            emission_scale_trils = np.linalg.cholesky(emission_covariances)

    emission_dist = \
    tfp.distributions.MultivariateNormalTriL(loc=emission_means,
                                             scale_tril=emission_scale_trils)

    return HMM(num_states, initial_dist, transition_dist, emission_dist)


def initialize_gaussian_hmm(rng, num_states, data, **kwargs):
    """Helper function to initialize a Gaussian HMM from the data.
    
    Picks random data points as the means, and sets the covariance to a fraction of
    the marginal covariance.

    Args:
        rng (jax.random.PRNGKey): JAX PRNG Key.
        num_states (int): The number of discrete states in the HMM.
        data (Array): The observed time series data array ``(timesteps, obs_dim)``.
        **kwargs: Additional keyword arguments for `make_gaussian_hmm` function.

    Returns:
        gaussian_hmm [HMM]: The initialized Gaussian HMM object.
    """

    # Pick random data points as the means
    num_timesteps, emission_dim = data.shape
    inds = jr.choice(rng, num_timesteps, shape=(num_states,), replace=False)
    means = data[inds]

    # from sklearn.cluster import KMeans
    # km = KMeans(num_states)
    # km.fit(data)
    # means = km.cluster_centers_

    # Set the covariance to a fraction of the marginal covariance
    cov = np.cov(data, rowvar=False)
    scale_tril = np.tile(np.linalg.cholesky(cov) / num_states, (num_states, 1, 1))

    return make_gaussian_hmm(
        num_states, emission_dim,
        emission_means=means,
        emission_scale_trils=scale_tril,
        **kwargs)

# Poisson HMM
def make_poisson_hmm(num_states: int,
                     emission_dim: int,
                     initial_state_probs: Array=None,
                     initial_state_logits: Array=None,
                     transition_matrix: Array=None,
                     transition_logits: Array=None,
                     emission_log_rates: Array=None): 
    """Helper function to create an HMM with Poisson observations.

    Args:
        num_states (int): Number of discrete states for the HMM.
        emission_dim (int): Dimension of emissions.
        initial_state_probs (Array, optional): Specify the initial state probabilities ``(num_states,)``.
            Defaults to None.
        initial_state_logits (Array, optional): Specify the initial state logits ``(num_states,)``.
            Defaults to None.
        transition_matrix (Array, optional): Specify the transition matrix. ``(num_states, num_states)``.
            Defaults to None.
        transition_logits (Array, optional): Specify the transition logits. ``(num_states, num_states)``.
            Defaults to None.
        emission_log_rates (Array, optional): Specify the log rates of emissions ``(num_states, emission_dim)``.
            Defaults to None.

    Returns:
        poisson_hmm (HMM): An initialized Poisson HMM.
    """

    # Initialize the basics
    initial_dist, transition_dist = \
        _make_standard_hmm(num_states,
                           initial_state_probs=initial_state_probs,
                           initial_state_logits=initial_state_logits,
                           transition_matrix=transition_matrix,
                           transition_logits=transition_logits)

    # Initialize the Gaussian emissions
    if emission_log_rates is None:
        emission_log_rates = np.zeros((num_states, emission_dim))

    emissions_dist = tfp.distributions.Independent(
        tfp.distributions.Poisson(log_rate=emission_log_rates),
        reinterpreted_batch_ndims=1,
    )

    return HMM(num_states, initial_dist, transition_dist, emissions_dist)


def initialize_poisson_hmm(rng: jr.PRNGKey, num_states: int, data: Array, **kwargs):
    """Initialize a Poisson HMM from the data.

    Args:
        rng (jax.random.PRNGKey): JAX PRNG Key.
        num_states (int): The number of discrete states for the HMM.
        data (Array): The observed time series data array ``(timesteps, obs_dim)``.
        **kwargs: Additional keyword arguments for ``make_poisson_hmm`` function.

    Returns:
        poisson_hmm (HMM): The initialized Poisson HMM object.
    """

    # Pick random data points as the means
    num_timesteps, emission_dim = data.shape
    assignments = jr.choice(rng, num_states, shape=(num_timesteps,))
    rates = np.row_stack(
        [data[assignments == k].mean(axis=0) for k in range(num_states)]
    )

    return make_poisson_hmm(
        num_states, emission_dim,
        emission_log_rates=np.log(rates),
        **kwargs)
