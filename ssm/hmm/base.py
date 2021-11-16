"""Module defining base model behavior for Hidden Markov Models (HMMs).
"""
from dataclasses import dataclass
from enum import auto

import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class

from ssm.base import SSM
from ssm.inference.em import em
from ssm.utils import Verbosity, auto_batch, one_hot, ensure_has_batch_dim

import ssm.hmm.initial as initial
import ssm.hmm.transitions as transitions
import ssm.hmm.emissions as emissions
from ssm.hmm.posterior import StationaryHMMPosterior



@register_pytree_node_class
class HMM(SSM):
    """The Hidden Markov Model base class.
    """
    def __init__(self, num_states: int,
                 initial_condition: initial.InitialCondition,
                 transitions: transitions.Transitions,
                 emissions: emissions.Emissions,
                 ):
        """The HMM base class.

        Args:
            num_states (int): number of discrete states
            initial_condition (initial.InitialCondition):
                initial condition object defining :math:`p(z_1)`
            transitions (transitions.Transitions):
                transitions object defining :math:`p(z_t|z_{t-1})`
            emissions (emissions.Emissions):
                emissions ojbect defining :math:`p(x_t|z_t)`
        """
        # Store these as private class variables
        self._num_states = num_states
        self._initial_condition = initial_condition
        self._transitions = transitions
        self._emissions = emissions

    @property
    def transition_matrix(self):
        return self._transitions.transition_matrix

    @property
    def num_states(self):
        return self._num_states

    @property
    def emissions_shape(self):
        return self._emissions.emissions_shape

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # We have to be a little fancy since this classmethod
        # is inherited by subclasses with different constructors.
        obj = object.__new__(cls)
        HMM.__init__(obj, aux_data, *children)
        return obj

    def initial_distribution(self, covariates=None, metadata=None):
        return self._initial_condition.distribution(covariates=covariates, metadata=metadata)

    def dynamics_distribution(self, state, covariates=None, metadata=None):
        return self._transitions.distribution(state, covariates=covariates, metadata=metadata)

    def emissions_distribution(self, state, covariates=None, metadata=None):
        return self._emissions.distribution(state, covariates=covariates, metadata=metadata)


    ### Methods for posterior inference
    @ensure_has_batch_dim()
    def initialize(self,
                   key: jr.PRNGKey,
                   data: np.ndarray,
                   covariates: np.ndarray=None,
                   metadata: np.ndarray=None,
                   method: str="kmeans") -> None:
        r"""Initialize the model parameters by performing an M-step with state assignments
        determined by the specified method (random or kmeans).

        Args:
            dataset (np.ndarray): array of observed data
                of shape :math:`(\text{batch} , \text{num\_timesteps} , \text{emissions\_dim})`
            key (jr.PRNGKey): random seed
            method (str, optional): state assignment method.
                One of "random" or "kmeans". Defaults to "kmeans".

        Raises:
            ValueError: when initialize method is not recognized
        """
        # initialize assignments and perform one M-step
        num_states = self._num_states
        if method.lower() == "random":
            # randomly assign datapoints to clusters
            # TODO: use self.emissions_shape
            assignments = jr.choice(key, self._num_states, data.shape[:-1])

        elif method.lower() == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(num_states)
            # TODO: use self.emissions_shape
            flat_dataset = data.reshape(-1, data.shape[-1])
            assignments = km.fit_predict(flat_dataset).reshape(data.shape[:-1])

        else:
            raise ValueError(f"Invalid initialize method: {method}.")

        # Make a dummy posterior that just exposes expected_states
        @dataclass
        class DummyPosterior:
            expected_states: np.ndarray
        dummy_posteriors = DummyPosterior(one_hot(assignments, self._num_states))

        # Do one m-step with the dummy posteriors
        self._emissions.m_step(data, dummy_posteriors)

    ### EM: Operates on batches of data (aka datasets) and posteriors
    @auto_batch(batched_args=("data", "posterior", "covariates", "metadata"))
    def marginal_likelihood(self, data, posterior=None, covariates=None, metadata=None):
        if posterior is None:
            posterior = self.e_step(data, covariates=covariates, metadata=metadata)

        return posterior.log_normalizer

    @auto_batch(batched_args=("data", "covariates", "metadata"))
    def e_step(self, data, covariates=None, metadata=None):
        return StationaryHMMPosterior.infer(
            self._initial_condition.log_initial_probs(data, covariates=covariates, metadata=metadata),
            self._emissions.log_likelihoods(data, covariates=covariates, metadata=metadata),
            self._transitions.log_transition_matrices(data, covariates=covariates, metadata=metadata))

    @ensure_has_batch_dim()
    def m_step(self, data, posterior, covariates=None, metadata=None):
        self._initial_condition.m_step(data, posterior, covariates=covariates, metadata=metadata)
        self._transitions.m_step(data, posterior, covariates=covariates, metadata=metadata)
        self._emissions.m_step(data, posterior, covariates=covariates, metadata=metadata)

    @ensure_has_batch_dim()
    def fit(self, data: np.ndarray,
            covariates=None,
            metadata=None,
            method: str="em",
            num_iters: int=100,
            tol: float=1e-4,
            initialization_method: str="kmeans",
            key: jr.PRNGKey=None,
            verbosity: Verbosity=Verbosity.DEBUG):
        r"""Fit the HMM to a dataset using the specified method and initialization.

        Args:
            dataset (np.ndarray): observed data
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            method (str, optional): model fit method.
                Must be one of ["em"]. Defaults to "em".
            num_iters (int, optional): number of fit iterations.
                Defaults to 100.
            tol (float, optional): tolerance in log probability to determine convergence.
                Defaults to 1e-4.
            initialization_method (str, optional): method to initialize latent states.
                Defaults to "kmeans".
            key (jr.PRNGKey, optional): Random seed.
                Defaults to None.
            verbosity (Verbosity, optional): print verbosity.
                Defaults to Verbosity.DEBUG.

        Raises:
            ValueError: if fit method is not reocgnized

        Returns:
            log_probs (np.ndarray): log probabilities at each fit iteration
            model (HMM): the fitted model
            posteriors (StationaryHMMPosterior): the fitted posteriors
        """
        model = self

        if initialization_method is not None:
            if verbosity >= Verbosity.LOUD : print("Initializing...")
            self.initialize(key, data, method=initialization_method)
            if verbosity >= Verbosity.LOUD: print("Done.", flush=True)

        if method == "em":
            log_probs, model, posteriors = em(
                model, data, num_iters=num_iters, tol=tol, verbosity=verbosity,
                covariates=covariates, metadata=metadata
            )

        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return log_probs, model, posteriors

    def __repr__(self):
        return f"<ssm.hmm.{type(self).__name__} num_states={self.num_states} " \
               f"emissions_shape={self.emissions_shape}>"
