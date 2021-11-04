import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from ssm.inference.laplace_em import _laplace_e_step, laplace_em
from ssm.base import SSM

from ssm.utils import Verbosity, format_dataset

import ssm.lds.initial as initial
import ssm.lds.dynamics as dynamics
import ssm.lds.emissions as emissions


@register_pytree_node_class
class LDS(SSM):
    def __init__(self,
                 initial_condition: initial.InitialCondition,
                 dynamics: dynamics.Dynamics,
                 emissions: emissions.Emissions,
                 ):
        self._initial_condition = initial_condition
        self._dynamics = dynamics
        self._emissions = emissions

    @property
    def latent_dim(self):
        return self._emissions.weights.shape[-1]

    @property
    def emissions_dim(self):
        return self._emissions.weights.shape[-2]

    @property
    def initial_mean(self):
        return self._initial_condition.mean

    @property
    def initial_covariance(self):
        return self._initial_condition._initial_distribution.covariance()

    @property
    def dynamics_matrix(self):
        return self._dynamics.weights

    @property
    def dynamics_bias(self):
        return self._dynamics.bias

    @property
    def dynamics_noise_covariance(self):
        Q_sqrt = self._dynamics.scale_tril
        return Q_sqrt @ Q_sqrt.T

    @property
    def emissions_matrix(self):
        return self._emissions.weights

    @property
    def emissions_bias(self):
        return self._emissions.bias

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data, *children)

    def initial_distribution(self):
        return self._initial_condition.distribution()

    def dynamics_distribution(self, state):
        return self._dynamics.distribution(state)

    def emissions_distribution(self, state):
        return self._emissions.distribution(state)

    ### Methods for posterior inference
    def approximate_posterior(self, data, initial_states=None):
        return _laplace_e_step(self, data, initial_states)

    def m_step(self, dataset, posteriors, rng=None):
        # self._initial_condition.m_step(dataset, posteriors)  # TODO initial dist needs prior
        self._dynamics.m_step(dataset, posteriors)
        self._emissions.m_step(dataset, posteriors, rng=rng)

    @format_dataset
    def fit(self, dataset, method="laplace_em", rng=None, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG):
        model = self
        if method == "laplace_em":
            elbos, lds, posteriors = laplace_em(rng, model, dataset, num_iters=num_iters, tol=tol)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return elbos, lds, posteriors