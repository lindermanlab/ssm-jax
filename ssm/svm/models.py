"""
Implementations of various stochastic volatility models.
"""

import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class
from ssm.base import SSM
from ssm.lds.initial import StandardInitialCondition
from ssm.lds.dynamics import Dynamics, StationaryDynamics
from ssm.lds.emissions import Emissions
import ssm.utils as utils

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


@register_pytree_node_class
class UnivariateSVM(SSM):
    """
    Follows the definition and nomenclature in Scibor & Wood 2021.
    """

    def __init__(self,
                 log_sigma: float = np.log(1.0),
                 invsig_phi: float = utils.inverse_sigmoid(0.9),
                 mu: float = 2.0,
                 init_mean: float = 0.0,
                 transition_log_scale_tril_multiplier: float = np.log(1.0),
                 emission_log_scale_tril_multiplier: float = np.log(1.0)):
        """

        Args:
            log_sigma:
            invsig_phi:
            mu:
            init_mean:
        """

        # We are only considering the univariate case.
        self.num_states = 1

        # Inscribe the parameters.
        self.log_sigma = log_sigma
        self.invsig_phi = invsig_phi
        self.mu = mu
        self.init_mean = init_mean
        self.transition_log_scale_tril_multiplier = transition_log_scale_tril_multiplier
        self.emission_log_scale_tril_multiplier = emission_log_scale_tril_multiplier

        # The initial condition is a Gaussian with a specific variance.
        initial_scale_tril = np.sqrt(np.square(np.exp(log_sigma))) / (1 - np.square(utils.sigmoid(invsig_phi)))
        self._initial_condition = StandardInitialCondition(initial_mean=init_mean,
                                                           initial_scale_tril=initial_scale_tril, )

        # Initialize the SVM transition model.
        # This is a normal distribution with the mean equal to an affine function of current state.
        affine_bias = self.mu * (1.0 - utils.sigmoid(self.invsig_phi))
        affine_weight = self.utils.sigmoid(self.invsig_phi)
        self._dynamics = StationaryDynamics(weights=affine_weight,
                                            bias=affine_bias,
                                            scale_tril=np.exp(log_sigma))

        # Initialize the SVM emission distribution.
        self._emissions = SVMEmission(emission_log_scale_tril_multiplier)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.__init__(aux_data, *children)

    @property
    def emissions_shape(self):
        """
        Returns the shape of a single emission, :math:`y_t`.

        Returns:
            A tuple or tree of tuples giving the emission shape(s).
        """
        return 1,

    def initial_distribution(self,
                             covariates=None,
                             metadata=None):
        """
        Call through to the initial distribution.

        Args:
            covariates:
            metadata:

        Returns:

        """
        return self._initial_condition.distribution(covariates, metadata)

    def dynamics_distribution(self,
                              state,
                              covariates=None,
                              metadata=None):
        """
        Call through to the dyanmics distribution.

        Args:
            state:
            covariates:
            metadata:

        Returns:

        """
        return self._dynamics.distribution(state, covariates, metadata)

    def emissions_distribution(self,
                               state,
                               covariates=None,
                               metadata=None):
        """
        Call through to the emissions distribution.

        Args:
            state:
            covariates:
            metadata:

        Returns:

        """
        return self._emissions.distribution(state, covariates, metadata)

    def m_step(self, dataset, posteriors):
        r"""
        No closed form m-step exists.
        Args:
            dataset:
            posteriors:

        Returns:

        """
        raise NotImplementedError


@register_pytree_node_class
class SVMEmission(Emissions):
    """
    The SVM emission distribution is a zero-mean Gaussian with a scale determined as a
    function of the current state.
    """

    def __init__(self,
                 emission_log_scale_tril_multiplier: float = np.log(1.0)):
        r"""

        Args:
            emission_log_scale_tril_multiplier (float):
                Multiplier for the scale (changes the variance of the `\epsilon` noise R.V.).
        """
        self.emission_log_scale_tril_multiplier = emission_log_scale_tril_multiplier

    def tree_flatten(self):
        children = (self.emission_log_scale_tril_multiplier, )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def distribution(self, state, covariates=None, metadata=None):
        """

        Args:
            state:
            covariates:
            metadata:

        Returns:

        """

        assert covariates is None, "Covariates are not provisioned under the original SVM."
        assert metadata is None, "Metadata is not provisioned under the original SVM."

        mean = 0.0  # Emission distribution is zero mean.
        scale = self.emission_log_scale_tril_multiplier * np.sqrt(np.exp(state / 2.0))  # Scale is defined conditioned on state.
        dist = tfd.Normal(mean, scale)
        return dist

    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               num_samples=1,
               key=None):
        r"""
        No closed-form m-step exists.
        Args:
            data:
            posterior:
            covariates:
            metadata:
            num_samples:
            key:

        Returns:

        """
        raise NotImplementedError


# @register_pytree_node_class
# class SVMDynamics(Dynamics):
#     """
#     Basic dynamics model for LDS.
#     """
#     def __init__(self,
#                  log_sigma: float,
#                  invsig_phi: float,
#                  mu: float,
#                  transition_log_scale_tril_multiplier: float) -> None:
#         super().__init__()
#
#         self.log_sigma = log_sigma
#         self.invsig_phi = invsig_phi
#         self.mu = mu
#         self.transition_log_scale_tril_multiplier = transition_log_scale_tril_multiplier
#
#     def tree_flatten(self):
#         children = (self.log_sigma, self.invsig_phi, self.mu)
#         aux_data = None
#         return children, aux_data
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)
#
#     @property
#     def scale(self):
#         return np.exp(self.transition_log_scale_tril_multiplier)
#
#     def distribution(self, state, covariates=None, metadata=None):
#         r"""
#         The transition distribution if defined as Gaussian with a mean determined by two terms.
#         There is an exponential decay term and an autoregressive term.
#
#         Args:
#             state:
#             covariates:     Must be `None`.
#             metadata:       Must be `None`.
#
#         Returns:
#
#         """
#
#         assert covariates is None, "Covariates are not provisioned under the original SVM."
#         assert metadata is None, "Metadata is not provisioned under the original SVM."
#
#         decay = self.mu * (1.0 - utils.sigmoid(self.invsig_phi))
#         autoregressive = utils.sigmoid(self.invsig_phi) * state
#         mean = decay + autoregressive
#
#         # The scale of the process is constant.
#         scale = np.exp(self.log_sigma)
#
#         # Build the distribution.
#         dist = tfd.Normal(mean, scale)
#
#         return dist
