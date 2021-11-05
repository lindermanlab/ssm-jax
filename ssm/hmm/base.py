"""
HMM Model Classes
=================

Module defining model behavior for Hidden Markov Models (HMMs).
"""
from typing import Any
from dataclasses import dataclass
Array = Any

import jax.random as jr
from jax.tree_util import register_pytree_node_class

from ssm.base import SSM
from ssm.inference.em import em
from ssm.utils import Verbosity, format_dataset, one_hot

import ssm.hmm.initial as initial
import ssm.hmm.transitions as transitions
import ssm.hmm.emissions as emissions
from ssm.hmm.posterior import StationaryHMMPosterior

@register_pytree_node_class
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

    @property
    def transition_matrix(self):
        return self._transitions.transition_matrix

    @property
    def num_states(self):
        return self._num_states

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

        # Make a dummy posterior that just exposes expected_states
        @dataclass
        class DummyPosterior:
            expected_states: Array
        dummy_posteriors = DummyPosterior(one_hot(assignments, self._num_states))

        # Do one m-step with the dummy posteriors
        self._emissions.m_step(dataset, dummy_posteriors)

    def infer_posterior(self, data):
        return StationaryHMMPosterior.infer(self._initial_condition.log_probs(data),
                                            self._emissions.log_probs(data),
                                            self._transitions.log_probs(data))

    def marginal_likelihood(self, data, posterior=None):
        if posterior is None:
            posterior = self.infer_posterior(data)

        # dummy_states = np.zeros(data.shape[0], dtype=int)
        # return self.log_probability(dummy_states, data) - posterior.log_prob(dummy_states)
        return posterior.log_normalizer

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
