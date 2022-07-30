import jax
from jax import grad, lax, vmap, random, jit
from jax.interpreters.masking import parse_spec
from jax.scipy.linalg import solve_triangular
from jax.tree_util import register_pytree_node_class
import jax.numpy as np
import jax.random as jr

from functools import partial

import ssm
from ssm.base import SSM
from ssm.lds.base import LDS
from ssm.lds.initial import StandardInitialCondition
from ssm.lds.dynamics import StationaryDynamics
from ssm.utils import Verbosity, ensure_has_batch_dim, auto_batch, \
    random_rotation, tree_get, tree_concatenate

from ssm.inference.deep_vi import deep_variational_inference
from ssm.deep_lds.posterior import LDSSVAEPosterior, DKFPosterior, DeepAutoregressivePosterior
from ssm.nn_util import GaussianNetworkFullMeanParams, BiRNNMeanParams, GaussianNetworkDiag

# For convenience
import importlib
importlib.reload(ssm.inference.deep_vi)

# To keep it really really simple, we don't even have to write ANYTHING
# Except for the fit function
@register_pytree_node_class
class DeepLDS(LDS):

    def __init__(self, num_latent_dims,
                 num_emission_dims,
                 emissions,
                 initial_state_mean: np.ndarray=None,
                 initial_state_scale_tril: np.ndarray=None,
                 dynamics_weights: np.ndarray=None,
                 dynamics_bias: np.ndarray=None,
                 dynamics_scale_tril: np.ndarray=None,
                 seed: jr.PRNGKey=None
                 ):
        
        if initial_state_mean is None:
            initial_state_mean = np.zeros(num_latent_dims)

        if initial_state_scale_tril is None:
            initial_state_scale_tril = np.eye(num_latent_dims)

        if dynamics_weights is None:
            seed, rng = jr.split(seed, 2)
            dynamics_weights = random_rotation(rng, num_latent_dims, theta=np.pi/20)

        if dynamics_bias is None:
            dynamics_bias = np.zeros(num_latent_dims)

        if dynamics_scale_tril is None:
            dynamics_scale_tril = 0.1**2 * np.eye(num_latent_dims)

        initial_condition = StandardInitialCondition(initial_mean=initial_state_mean,
                                                     initial_scale_tril=initial_state_scale_tril)
        transitions = StationaryDynamics(weights=dynamics_weights,
                                         bias=dynamics_bias,
                                         scale_tril=dynamics_scale_tril)
        
        super(DeepLDS, self).__init__(initial_condition, transitions, emissions)

    @property
    def latent_dim(self):
        return self._emissions.latent_dim

    @property
    def emissions_network(self):
        return self._emissions
    
    @property
    def emissions_shape(self):
        return self._emissions.emissions_shape

    def emissions_distribution(self, state, 
                               covariates=None, metadata=None, params=None):
        return self._emissions.distribution(state, params=params,
            covariates=covariates, metadata=metadata)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(*children)
        return obj

    # A temporary method
    def get_parameters(self):
        return (
            self.initial_mean,
            self.initial_covariance,
            self.dynamics_matrix,
            self.dynamics_bias,
            self.dynamics_noise_covariance
        )

    # This is identical to Collin's original code for LDS
    # Except that it returns  
    @ensure_has_batch_dim()
    def m_step(self,
               data: np.ndarray,
               posterior,
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
        return self

    # Ignore the covariates for now for simplicity
    def _dynamics_log_likelihood(self, states):
        lp = 0
        # Get the first timestep probability
        initial_state = tree_get(states, 0)
        lp += self.initial_distribution().log_prob(initial_state)

        def _step(carry, state):
            prev_state, lp = carry
            lp += self.dynamics_distribution(prev_state).log_prob(state)
            return (state, lp), None

        (_, lp), _ = lax.scan(_step, (initial_state, lp),
                                tree_get(states, slice(1, None)))
        return lp

    def _emissions_log_likelihood(self, states, data, 
                                  covariates=None, metadata=None, params=None):
        def _emissions_ll_single(state, emission, covariates):
            return self.emissions_distribution(state, 
            covariates=covariates, metadata=metadata, params=params).log_prob(emission)      

        return vmap(_emissions_ll_single)(states, data, covariates).sum()

    @auto_batch(batched_args=("states", "data", "covariates", "metadata"))
    def log_probability(self,
                        states,
                        data, 
                        covariates=None,
                        metadata=None,
                        params=None):
        return self._dynamics_log_likelihood(states) \
             + self._emissions_log_likelihood(states, data, params=params,
                    covariates=covariates, metadata=metadata)

    # Assume here that the posterior is for a single data 
    # because of auto-batching
    # We'll ignore the covariates for now for simplicity
    def _posterior_cross_entropy_to_dynamics(self, posterior):

        m1 = self.initial_mean
        Q1 = self.initial_covariance
        A = self.dynamics_matrix
        b = self.dynamics_bias
        Q = self.dynamics_noise_covariance
        T, D, _ = posterior.precision_diag_blocks.shape

        # diagonal blocks of precision matrix
        J = np.zeros((T, D, D))
        J = J.at[0].add(np.linalg.inv(Q1))
        J = J.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J = J.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        L = -np.linalg.solve(Q, A)
        L = np.tile(L[None, :, :], (T-1, 1, 1))

        Ex = posterior.expected_states
        ExxT = posterior.expected_states_squared
        ExnxT = posterior.expected_states_next_states
        Sigmatt = ExxT - np.einsum("ti,tj->tij", Ex, Ex)
        Sigmantt = ExnxT - np.einsum("ti,tj->tji", Ex[:-1], Ex[1:])

        cross_entropy = -self._dynamics_log_likelihood(Ex)
        cross_entropy += 0.5 * np.einsum("tij,tij->", J, Sigmatt) 
        cross_entropy += np.einsum("tij,tij->", L, Sigmantt)
        return cross_entropy

    @auto_batch(batched_args=("key", "data", "posterior", "covariates", "metadata"))
    def _elbo_cf(self,
             key,
             data,
             posterior,
             covariates=None,
             metadata=None,
             num_samples=1,
             params=None):
        # Assumeing that both the dynamics and the posterior are MVN block tridiags
        # We can split the ELBO differently and obtain a lower variance estimate
        # for the elbo

        post_params, dec_params = params

        def _sample_emissions_log_likelihood(key):
            state = posterior.sample(seed=key, params=post_params)
            return self._emissions_log_likelihood(state, data, 
                covariates=covariates, metadata=metadata, params=dec_params)

        ell = vmap(_sample_emissions_log_likelihood)(jr.split(key, num_samples)).mean(axis=0)
        cross_entropy = self._posterior_cross_entropy_to_dynamics(posterior)
        q_entropy = posterior.entropy()
        elbo = ell - cross_entropy + q_entropy
        return elbo

    @auto_batch(batched_args=("key", "data", "posterior", "covariates", "metadata"))
    def _elbo_sample(self,
             key,
             data,
             posterior,
             covariates=None,
             metadata=None,
             num_samples=1,
             params=None):
        """
        Same as the default elbo for ssm's
        """

        post_params, dec_params = params

        def _elbo_single(_key):
            sample = posterior.sample(seed=_key, params=post_params)
            return self.log_probability(sample, data, covariates, metadata, params=dec_params) \
                   -posterior.log_prob(sample, params=post_params)

        elbos = vmap(_elbo_single)(jr.split(key, num_samples))
        return np.mean(elbos)

    @ensure_has_batch_dim()
    def fit(self,
            key: jr.PRNGKey,
            data: np.ndarray,
            covariates=None,
            metadata=None,
            method: str="svae",
            num_iters: int=100,
            tol: float=1e-4,
            verbosity: Verbosity=Verbosity.DEBUG,
            recognition_model_class=None, 
            learning_rate=1e-3,
            recognition_only=False,
            init_emissions_params=None,
            sample_kl=False,
            # Addition parameters can include information about the
            # recognition network architecture and the specifics of training
            **params
            ):
        """
        Notably, returns the rec net as well as the model
        """
        # Just initialize the posterior since the model will be
        # updated on the first m-step.
        N, T, D = data.shape

        autoregressive_posterior = False

        if method == "svae":
            posterior_class = LDSSVAEPosterior
            default_recognition_model_class = GaussianNetworkFullMeanParams
        elif method in ["dkf", "cdkf"]:
            posterior_class = DKFPosterior
            default_recognition_model_class = BiRNNMeanParams
        elif method in ["planet"]:
            posterior_class = DeepAutoregressivePosterior
            default_recognition_model_class = GaussianNetworkDiag
            sample_kl = True
            autoregressive_posterior = True
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        # Whether or not to use the sampling approximation for the KL divergence
        if sample_kl:
            print("Using sampling approximation for KL!")
            DeepLDS.elbo = self._elbo_sample
        else:
            DeepLDS.elbo = self._elbo_cf

        posterior = posterior_class.initialize(
                self, data, covariates=covariates, metadata=metadata, **params)
        rec_net = (recognition_model_class or default_recognition_model_class)\
            .from_params(self.latent_dim, input_dim=D, **params)

        bounds, model, posterior = deep_variational_inference(
                key, self, data, rec_net, posterior, 
                covariates=covariates, metadata=metadata,
                num_iters=num_iters, learning_rate=learning_rate,
                tol=tol, verbosity=verbosity, 
                recognition_only=recognition_only,
                autoregressive_posterior=autoregressive_posterior,
                init_emissions_params=init_emissions_params, **params)

        return bounds, model, posterior
