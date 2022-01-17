"""
Implementations of various stochastic volatility models.
"""
import inspect

import jax
import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class
from ssm.base import SSM
from ssm.lds.initial import StandardInitialCondition
from ssm.lds.dynamics import Dynamics, StationaryDynamics
from ssm.lds.emissions import Emissions
import ssm.utils as utils
from ssm.utils import Verbosity, random_rotation, make_named_tuple, ensure_has_batch_dim, auto_batch
from ssm.inference.em import em
from ssm.inference.laplace_em import laplace_em
from jax.flatten_util import ravel_pytree
from jax import tree_util, vmap

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.distributions import MultivariateNormalBlockTridiag
SVMPosterior = MultivariateNormalBlockTridiag


@register_pytree_node_class
class SVM(SSM):
    """
    Follows the definition and nomenclature in Naesseth et al 2017.
    """

    def __init__(self,
                 num_latent_dims: int = 1,
                 num_emission_dims: int = 1,

                 mu: np.ndarray = np.asarray([0.0]),

                 # invsig_phi = 0utils.inverse_sigmoid(0.9).
                 invsig_phi: np.ndarray = np.asarray([2.197]),

                 # log_Q = np.log(1.0)
                 log_Q: np.ndarray = np.asarray([[0.0]]),

                 # log_beta = np.log(0.1)
                 log_beta: np.ndarray = np.asarray([-2.303]),

                 initial_condition=None,
                 dynamics=None,
                 emissions=None,

                 seed: jr.PRNGKey = None):
        """

        Args:
            num_latent_dims:
            num_emission_dims:
            mu:
            invsig_phi:
            log_Q:
            log_beta:
            seed:
        """

        # We are only considering the univariate case.
        self.latent_dim = num_latent_dims
        self.emission_dims = num_emission_dims

        # Inscribe the parameters.
        self.log_Q = log_Q
        self.invsig_phi = invsig_phi
        self.mu = mu
        self.log_beta = log_beta

        # Check the input shapes.
        assert self.invsig_phi.shape == (self.latent_dim, )
        # assert self.log_Q_diag.shape == (self.latent_dim, )
        # TODO - expand this to multivariate case.

        # The initial condition is a Gaussian with a specific variance.
        # initial_scale_tril = np.sqrt(np.square(np.exp(log_Q))) / (1 - np.square(utils.sigmoid(invsig_phi)))
        if initial_condition is None:
            self._initial_condition = StandardInitialCondition(initial_mean=self.mu,
                                                               initial_scale_tril=np.sqrt(np.exp(self.log_Q)), )
        else:
            self._initial_condition = initial_condition

        # Initialize the SVM transition model.
        # This is a normal distribution with the mean equal to an affine function of current state.
        if dynamics is None:
            affine_bias = self.mu * (1.0 - utils.sigmoid(self.invsig_phi))
            affine_weight = utils.sigmoid(self.invsig_phi) * np.eye(self.latent_dim)
            self._dynamics = StationaryDynamics(weights=affine_weight,
                                                bias=affine_bias,
                                                scale_tril=np.sqrt(np.exp(self.log_Q)))
        else:
            self._dynamics = dynamics

        # Initialize the SVM emission distribution.
        if emissions is None:
            self._emissions = SVMEmission(log_beta)
        else:
            self._emissions = emissions

        # Grab the parameter values.  This allows us to explicitly re-build the object.
        self._parameters = make_named_tuple(dict_in=locals(),
                                            keys=list(inspect.signature(self.__init__)._parameters.keys()),
                                            name=str(self.__class__.__name__) + 'Tuple')

    def tree_flatten(self):
        children = (self.mu,
                    self.invsig_phi,
                    self.log_Q,
                    self.log_beta,)
                    # self._initial_condition,
                    # self._dynamics,
                    # self._emissions)
        aux_data = (self.latent_dim, self.emission_dims)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

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

    @ensure_has_batch_dim()
    def m_step(self,
               data: np.ndarray,
               posterior: SVMPosterior,
               covariates=None,
               metadata=None,
               key: jr.PRNGKey=None):
        """Update the model in a (potentially approximate) M step.

        Updates the model in place.

        Args:
            data (np.ndarray): observed data with shape (B, T, D)
            posterior (LDSPosterior): LDS posterior object with leaf shapes (B, ...).
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            key (jr.PRNGKey, optional): random seed. Defaults to None.
        """
        # self._initial_condition.m_step(dataset, posteriors)  # TODO initial dist needs prior
        self._dynamics.m_step(data, posterior)
        self._emissions.m_step(data, posterior, key=key)

    @ensure_has_batch_dim()
    def fit(self,
            data: np.ndarray,
            covariates=None,
            metadata=None,
            method: str="em",
            key: jr.PRNGKey=None,
            num_iters: int=100,
            tol: float=1e-4,
            verbosity: Verbosity=Verbosity.DEBUG):
        r"""Fit the GaussianLDS to a dataset using the specified method.

        Note: because the observations are Gaussian, we can perform exact EM for a GaussianEM
        (i.e. the model is conjugate).

        Args:
            data (np.ndarray): observed data
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            covariates (PyTreeDef, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTreeDef, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            method (str, optional): model fit method. Must be one of ["em", "laplace_em"].
                Defaults to "em".
            key (jr.PRNGKey, optional): Random seed.
                Defaults to None.
            num_iters (int, optional): number of fit iterations.
                Defaults to 100.
            tol (float, optional): tolerance in log probability to determine convergence.
                Defaults to 1e-4.
            verbosity (Verbosity, optional): print verbosity.
                Defaults to Verbosity.DEBUG.

        Raises:
            ValueError: if fit method is not reocgnized

        Returns:
            elbos (np.ndarray): elbos at each fit iteration
            model (LDS): the fitted model
            posteriors (LDSPosterior): the fitted posteriors
        """
        model = self
        kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

        if method == "em":
            elbos, lds, posteriors = em(model, data, **kwargs)
        elif method == "laplace_em":
            if key is None:
                raise ValueError("Laplace EM requires a PRNGKey. Please provide an rng to fit.")
            elbos, lds, posteriors = laplace_em(key, model, data, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return elbos, lds, posteriors


@register_pytree_node_class
class SVMEmission(Emissions):
    """
    The SVM emission distribution is a zero-mean Gaussian with a scale determined as a
    function of the current state.
    """

    def __init__(self,
                 log_beta: float):
        r"""

        Args:
            log_beta (float):
                Multiplier for the scale (changes the variance of the `\epsilon` noise R.V.).
        """
        self.log_beta = log_beta

    def tree_flatten(self):
        children = (self.log_beta, )
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

        mean = np.zeros_like(self.log_beta)  # Emission distribution is zero mean.
        scale = np.sqrt(np.exp(self.log_beta) * np.exp(state / 2.0))  # Scale is defined conditioned on state.
        dist = tfd.MultivariateNormalDiag(mean, scale)
        return dist

    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               num_samples=1,
               key=None):
        """
        NOTE - this mirrors the LDS->Emissions->PoissionLDS->m_step.

        Args:
            data:
            posterior:
            covariates:
            metadata:
            num_samples:
            key:

        Returns:

        """
        if key is None:
            raise ValueError("PRNGKey needed for generic m-step")

        # Draw samples of the latent states
        state_samples = posterior.sample(seed=key, sample_shape=(num_samples,))

        # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
        # NOTE - AW - changed to ravel `self` instead of `self._distribution`.
        flat_emissions_distribution, unravel = ravel_pytree(self)

        def _objective(flat_emissions_distribution):
            # TODO: Consider proximal gradient descent to counter sampling noise
            emissions_distribution = unravel(flat_emissions_distribution)

            def _lp_single(sample):
                if covariates is not None:
                    sample = np.concatenate([sample, covariates], axis=-1)
                return emissions_distribution.distribution(sample).log_prob(data) / data.size

            return -1 * np.mean(vmap(_lp_single)(state_samples))

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            flat_emissions_distribution,
            method="BFGS"  # TODO: consider L-BFGS?
        )

        optimized_distribution = unravel(optimize_results.x)

        # Inscribe the results.
        self.log_beta = optimized_distribution.log_beta


# p = 0
#
# # Initialize our true SVM model
# true_lds = UnivariateSVM()
#
# import warnings
#
# num_trials = 5
# time_bins = 200
#
# # catch annoying warnings of tfp Poisson sampling
# rng = jr.PRNGKey(0)
# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', category=UserWarning)
#     all_states, all_data = true_lds.sample(key=rng, num_steps=time_bins, num_samples=num_trials)
#
#
# sig_init = np.asarray([[np.log(0.1)]])
# emission_mult = np.asarray([np.log(0.1)])
# test_svm = UnivariateSVM(log_Q=sig_init, log_beta=emission_mult)
#
# print(test_svm.tree_flatten())
#
# rng = jr.PRNGKey(10)
#
# elbos, fitted_svm, posteriors = test_svm.fit(all_data, method="laplace_em", key=rng, num_iters=25)
#
# # with jax.disable_jit():
# #     elbos, fitted_svm, posteriors = test_svm.fit(all_data, method="laplace_em", key=rng, num_iters=25)
#
# print(elbos)
# print(fitted_svm.tree_flatten())
#
# p = 0


# @register_pytree_node_class
# class SVMDynamics(Dynamics):
#     """
#     Basic dynamics model for LDS.
#     """
#     def __init__(self,
#                  log_Q: float,
#                  invsig_phi: float,
#                  mu: float, ) -> None:
#         super().__init__()
#
#         self.log_Q = log_Q
#         self.invsig_phi = invsig_phi
#         self.mu = mu
#
#     def tree_flatten(self):
#         children = (self.log_Q, self.invsig_phi, self.mu)
#         aux_data = None
#         return children, aux_data
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)
#
#     @property
#     def scale(self):
#         return np.exp(self.log_Q)
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
#         scale = np.exp(self.log_Q)
#
#         # Build the distribution.
#         dist = tfd.Normal(mean, scale)
#
#         return dist
