"""
HMM Model Classes
=================

Module defining model behavior for Hidden Markov Models (HMMs).
"""
from typing import Any

from jax._src.numpy.lax_numpy import isin

Array = Any

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp

from ssm.base import SSM
from ssm.inference.em import em
from ssm.hmm.posterior import hmm_expected_states, HMMPosterior
from ssm.utils import Verbosity, format_dataset, one_hot

import ssm.hmm.initial as initial
import ssm.hmm.transitions as transitions
import ssm.hmm.emissions as emissions
from ssm.distributions.discrete_chain import StationaryDiscreteChain

class HMM(SSM):

    def __init__(self, num_states: int,
                 initial_condition: initial.InitialCondition,
                 transitions: transitions.Transitions,
                 emissions: emissions.Emissions,
                 ):
        # Store these as private class variables
        self._num_states = num_states
        self._initial_condition = initial_condition
        self._transitions = transitions
        self._emissions = emissions

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data, *children)

    def initial_distribution(self):
        return self._initial_condition.distribution()

    def dynamics_distribution(self, state):
        return self._transitions.distribution(state)

    def emissions_distribution(self, state):
        return self._emissions.distribution(state)

    @property
    def transition_matrix(self):
        return self._transitions.transition_matrix()

    ### Methods for posterior inference
    @format_dataset
    def initialize(self, dataset, key, method="kmeans"):
        """
        Initialize the model parameters by performing an M-step with state assignments
        determined by the specified method (random or kmeans).
        """
        # initialize assignments and perform one M-step
        num_states = self._num_states
        if method.lower() == "random":
            # randomly assign datapoints to clusters
            assignments = jr.choice(key, self._num_states, dataset.shape[:-1])

        elif method.lower() == "kmeans":
            # cluster the data with kmeans
            print("initializing with kmeans")
            from sklearn.cluster import KMeans
            km = KMeans(num_states)
            flat_dataset = dataset.reshape(-1, dataset.shape[-1])
            assignments = km.fit_predict(flat_dataset).reshape(dataset.shape[:-1])

        else:
            raise Exception("Observations.initialize: "
                "Invalid initialize method: {}".format(method))

        Ez = one_hot(assignments, self._num_states)
        dummy_posteriors = HMMPosterior(None, Ez, None)
        self._emissions.m_step(dataset, dummy_posteriors)

    def infer_posterior(self, data):
        ks = np.arange(self._num_states)
        initial_log_probs = self._initial_condition.distribution().log_prob(ks)
        transition_log_probs = vmap(
            lambda i: self._transitions.distribution(i).log_prob(ks)
            )(ks)
        emission_log_probs = vmap(
            lambda k: self._emissions.distribution(k).log_prob(data))(ks).T

        return StationaryDiscreteChain(
            initial_log_probs,
            emission_log_probs,
            transition_log_probs)

    def marginal_likelihood(self, data, posterior=None):
        if posterior is None:
            posterior = self.infer_posterior(data)

        dummy_states = np.zeros(data.shape[0], dtype=int)
        return self.log_probability(dummy_states, data) - posterior.log_prob(dummy_states)

    ### EM: Operates on batches of data (aka datasets) and posteriors
    def m_step(self, dataset, posteriors):
        self._initial_condition.m_step(dataset, posteriors)
        self._transitions.m_step(dataset, posteriors)
        self._emissions.m_step(dataset, posteriors)

    @format_dataset
    def fit(self, dataset,
            method="em",
            num_iters=100,
            tol=1e-4,
            initialization_method="kmeans",
            key=None,
            verbosity=Verbosity.DEBUG):
        """
        Fit the parameters of the HMM using the specified method.

        Args:

        dataset: see `help(HMM)` for details.

        method: specification of how to fit the data.  Must be one
        of the following strings:
        - em

        initialization_method: optional method name ("kmeans" or "random")
        indicating how to initialize the model before fitting.

        key: jax.PRNGKey for random initialization and/or fitting

        verbosity: specify how verbose the print-outs should be.  See
        `ssm.util.Verbosity`.
        """
        model = self
        kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

        if initialization_method is not None:
            if verbosity >= Verbosity.LOUD : print("Initializing...")
            self.initialize(dataset, key, method=initialization_method)
            if verbosity >= Verbosity.LOUD: print("Done.", flush=True)

        if method == "em":
            log_probs, model, posteriors = em(model, dataset, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return log_probs, model, posteriors


@register_pytree_node_class
class AutoregressiveHMM(HMM):
    """
    TODO
    """
    @property
    def emission_dim(self):
        return self._emission_distribution.data_dimension

    @property
    def num_lags(self):
        return self._emission_distribution.covariate_dimension // self._emission_distribution.data_dimension

    def log_probability(self, states, data, prev_emissions=None):
        """
        Computes the log joint probability of a set of states and data (observations).

        .. math::
            \log p(x, y) = \log p(x_1) + \sum_{t=1}^{T-1} \log p(x_{t+1} | x_t) + \sum_{t=1}^{T} \log p(y_t | x_t)

        Args:
            states: An array of latent states (:math:`x_{1:T}`).
            data: An array of the observed data (:math:`y_{1:T}`).

        Returns:
            lp:
                The joint log probability of the provided states and data.
        """
        if prev_emissions is None:
            prev_emissions = np.zeros((self.num_lags, self.emission_dim))

        lp = 0
        lp += self.initial_distribution().log_prob(states[0])
        lp += self.emissions_distribution(states[0]).log_prob(data[0], covariates=prev_emissions.ravel())

        def _step(carry, args):
            prev_state, prev_emissions, lp = carry
            state, emission = args
            lp += self.dynamics_distribution(prev_state).log_prob(state)
            lp += self.emissions_distribution(state).log_prob(emission, covariates=prev_emissions.ravel())
            new_prev_emissions = np.row_stack([prev_emissions[1:], emission])
            return (state, new_prev_emissions, lp), None

        initial_carry = (states[0], np.row_stack([prev_emissions[1:], data[0]]), lp)
        (_, _, lp), _ = lax.scan(_step, initial_carry, (states[1:], data[1:]))
        return lp


    def sample(self, key, num_steps: int, initial_state=None, num_samples=1, prev_emissions=None):
        """
        Sample from the joint distribution defined by the state space model.

        .. math::
            x, y \sim p(x, y)

        Args:
            key (PRNGKey): A JAX pseudorandom number generator key.
            num_steps (int): Number of steps for which to sample.
            initial_state: Optional state on which to condition the sampled trajectory.
                Default is None which samples the intial state from the initial distribution.
            prev_emissions: Optional initial emissions to start the autoregressive model.

        Returns:
            states: A ``(timesteps,)`` array of the state value across time (:math:`x`).
            emissions: A ``(timesteps, obs_dim)`` array of the observations across time (:math:`y`).

        """

        def _sample(key):
            if initial_state is None:
                key1, key = jr.split(key, 2)
                state = self.initial_distribution().sample(seed=key1)
            else:
                state = initial_state

            if prev_emissions is None:
                history = np.zeros((self.num_lags, self.emission_dim))
            else:
                history = prev_emissions

            def _step(carry, key):
                history, state = carry
                key1, key2 = jr.split(key, 2)
                emission = self.emissions_distribution(state).sample(seed=key1, covariates=history.ravel())
                next_state = self.dynamics_distribution(state).sample(seed=key2)
                next_history = np.row_stack([history[1:], emission])
                return (next_history, next_state), (state, emission)

            keys = jr.split(key, num_steps)
            _, (states, emissions) = lax.scan(_step, (history, state), keys)
            return states, emissions

        if num_samples > 1:
            batch_keys = jr.split(key, num_samples)
            states, emissions = vmap(_sample)(batch_keys)
        else:
            states, emissions = _sample(key)

        return states, emissions

    def _log_likelihoods(self, data: Array):
        def _compute_ll(x, y):
            ll = self._emission_distribution.log_prob(y, covariates=x.ravel())
            new_x = np.row_stack([x[1:], y])
            return new_x, ll

        _, log_likelihoods = lax.scan(_compute_ll, np.zeros((self.num_lags, self.emission_dim)), data)

        # Ignore likelihood of the first bit of data since we don't have a prefix
        log_likelihoods = log_likelihoods.at[:self.num_lags].set(0.0)
        return log_likelihoods
