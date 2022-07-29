import jax.numpy as np
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_leaves, register_pytree_node_class

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import auto_batch
from ssm.debug import scan
from ssm.nn_util import MLP, Identity, lie_params_to_constrained

from tensorflow_probability.substrates import jax as tfp

from dataclasses import dataclass

from typing import (NamedTuple, Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict, ClassVar)
import flax.linen as nn

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any

tfd = tfp.distributions

class MVNBlockTridiagPosterior(MultivariateNormalBlockTridiag):
    @classmethod
    def initialize(cls, svae, data, covariates=None, metadata=None, **kwargs):
        # Create a dummy object
        # Assumes that the data has a batch dimension! (prevents re-jitting)
        N = data.shape[0]
        T = data.shape[-2]
        D = svae.latent_dim
        precision_diag_blocks = np.zeros((N, T, D, D))
        precision_lower_diag_blocks = np.zeros((N, T-1, D, D))
        linear_potential = np.zeros((N, T, D))
        log_normalizer = np.ones((N,))
        filtered_precisions = np.zeros((N, T, D, D))
        filtered_linear_potentials = np.zeros((N, T, D))
        expected_states = np.zeros((N, T, D))
        expected_states_squared = np.zeros((N, T, D, D))
        expected_states_next_states = np.zeros((N, T-1, D, D))

        self = cls(precision_diag_blocks,
                 precision_lower_diag_blocks,
                 linear_potential,
                 log_normalizer,
                 filtered_precisions,
                 filtered_linear_potentials,
                 expected_states,
                 expected_states_squared,
                 expected_states_next_states)

        return self
        
    @property
    def emissions_shape(self):
        return (self.precision_diag_blocks.shape[-1],)

    # def log_prob(self, xs, params=None):
    #     return super().log_prob(xs)

    # def sample(self, key)

# No need to register because we are not adding additional data, just methods
class LDSSVAEPosterior(MVNBlockTridiagPosterior):
        
    @auto_batch(batched_args=("data", "potential", "covariates", "metadata", "key"))
    def update(self, lds, data, potential, covariates=None, metadata=None, key=None, **kwargs):
        # Shorthand names for parameters
        m1 = lds.initial_mean
        Q1 = lds.initial_covariance
        A = lds.dynamics_matrix
        b = lds.dynamics_bias
        Q = lds.dynamics_noise_covariance
        # From data
        J_obs, h_obs = potential
        seq_len = data.shape[0]
        latent_dim = J_obs.shape[-1]

        # diagonal blocks of precision matrix
        J_diag = J_obs  # from observations
        # Expand the diagonals into full covariance matrices
        # J_diag = J_obs[..., None] * np.eye(latent_dim)[None, ...]
        J_diag = J_diag.at[0].add(np.linalg.inv(Q1))
        J_diag = J_diag.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J_diag = J_diag.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        J_lower_diag = -np.linalg.solve(Q, A)
        J_lower_diag = np.tile(J_lower_diag[None, :, :], (seq_len - 1, 1, 1))

        # linear potential
        h = h_obs  # from observations
        h = h.at[0].add(np.linalg.solve(Q1, m1))
        h = h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
        h = h.at[1:].add(np.linalg.solve(Q, b))

        return LDSSVAEPosterior.infer(J_diag, J_lower_diag, h)

class DKFPosterior(MVNBlockTridiagPosterior):
    @auto_batch(batched_args=("data", "potential", "covariates", "metadata", "key"))
    def update(self, lds, data, potential, covariates=None, metadata=None, key=None):
        # From data
        J_obs, L_obs, h_obs = potential
        return DKFPosterior.infer(J_obs, L_obs, h_obs)

# This is a posterior object that stores a network and its parameters
@register_pytree_node_class
@dataclass
class DeepAutoregressivePosterior:

    NUM_SAMPLES : ClassVar[int] = 100

    dynamics : nn.Module = None
    inputs : Array = None
    output_single_dummy : Array = None
    _expected_states: Array = None
    _expected_states_squared: Array = None
    _expected_states_next_states: Array = None

    @classmethod
    def initialize(cls, model, data, **kwargs):
        N, T, D = data.shape
        output_dim = kwargs["dataset_params"]["num_latent_dims"]
        input_dim = kwargs["autoreg_posterior_architecture"]["latent_dim"]
        dynamics = build_deep_autoregressive_dynamics(output_dim, **kwargs)
        inputs = np.zeros((N, T, input_dim))
        return cls(dynamics, inputs, np.zeros((N, output_dim)), 
                   None, None, None)

    def tree_flatten(self):
        children = (self.dynamics,
                    self.inputs, 
                    self.output_single_dummy,
                    self._expected_states,
                    self._expected_states_squared,
                    self._expected_states_next_states)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def init(self, key):
        # Return the parameters of the GRU cell
        latent_dim = self.inputs.shape[-1]
        dummy_h = np.zeros((self.output_single_dummy.shape[-1],))
        dummy_u = np.zeros((latent_dim,))
        return self.dynamics.init(key, dummy_h, dummy_u)

    # Note: we have to assume that the this function doesn't know anything
    # about batch dimensions
    def log_prob(self, xs, params=None):

        def _log_prob_single(xs, us):
            def _log_prob_step(carry, i):
                h = carry
                x, u = xs[i], us[i]
                (cov, mean) = self.dynamics.apply(params, h, u)
                pred_dist = tfd.MultivariateNormalFullCovariance(loc=mean, 
                                                            covariance_matrix=cov)
                log_prob = pred_dist.log_prob(x)
                carry = x
                return carry, log_prob

            init = np.zeros_like(self.output_single_dummy)
            _, log_probs = scan(_log_prob_step, init, np.arange(xs.shape[0]))
            return np.sum(log_probs, axis=0)
            
        return _log_prob_single(xs, self.inputs)

    # This function should take care of all the potential batching
    def sample(self, seed, params=None):

        # Sample a single sequence
        def _sample_single(seed, inputs, output_dummy):
            def _sample_step(carry, input):
                key, h = carry
                key, new_key = jr.split(key)
                u = input
                (cov, mean) = self.dynamics.apply(params, h, u)
                sample = jr.multivariate_normal(key, mean, cov)
                carry = new_key, sample
                output = sample
                return carry, output

            init = np.zeros_like(output_dummy)
            _, sample = scan(_sample_step, (seed, init), inputs)
            return sample

        if len(self.inputs.shape) == 2:
            return _sample_single(seed, self.inputs, 
                self.output_single_dummy)
        else:
            # batch mode
            return vmap(_sample_single)(seed, self.inputs, 
                self.output_single_dummy)

    

    # Pass in the parameters for optimization
    @classmethod
    def compute_moments(cls, posterior, params):
        def _update_single(p):
            # Make some samples to help estimate the moments
            seed = jr.PRNGKey(0)
            samples = vmap(p.sample, in_axes=(0, None))\
                (jr.split(seed, cls.NUM_SAMPLES), params)
            # Save the moments
            p._expected_states = np.mean(samples, axis=0)
            p._expected_states_squared = np.mean(
                np.einsum("nti,ntj->ntij", samples, samples), axis=0)
            p._expected_states_next_states = np.mean(
                np.einsum("nti,ntj->ntij", samples[:,:-1], samples[:,1:]), axis=0)
            return p
        if len(posterior.inputs.shape) == 2:
            return _update_single(posterior)
        else:
            return vmap(_update_single)(posterior)

    @property
    def expected_states(self):
        return self._expected_states

    @property
    def expected_states_squared(self):
        return self._expected_states_squared

    @property
    def expected_states_next_states(self):
        return self._expected_states_next_states

    def mean(self):
        return self.expected_states

    def covariance(self):
        """
        NOTE: This computes the _marginal_ covariance Cov[x_t] for each t
        """
        Ex = self._expected_states
        ExxT = self._expected_states_squared
        return ExxT - np.einsum("...i,...j->...ij", Ex, Ex)

    @auto_batch(batched_args=("data", "potential", "covariates", "metadata", "key"))
    def update(self, lds, data, potential, 
               covariates=None, metadata=None, key=None):
        _, inputs = potential
        return DeepAutoregressivePosterior(self.dynamics, inputs, 
                                           # This is such a big hack
                                           np.zeros((self.output_single_dummy.shape[-1],)),
                                           None, None, None)

    # Because auto_batch assumes emissions shape...
    @property
    def emissions_shape(self):
        return (self.output_single_dummy.shape[-1],)

# This is a hack for making our custom nn.Module a pytree...!
def build_deep_autoregressive_dynamics(output_dim, **kwargs):

    params = kwargs["autoreg_posterior_architecture"]

    # Define a GRU cell here
    rnn_cell = nn.GRUCell()
    trunk_fn = Identity(output_dim)
    head_mean_fn = MLP(params["out_mean_features"] + [output_dim,])
    
    init_cov_bias = params["cov_init"]
    head_cov_fn = MLP(
        params["out_var_features"] + [output_dim * (output_dim + 1) // 2,], 
        nn.initializers.zeros, 
        nn.initializers.constant(init_cov_bias))
    
    eps = params.get("eps") or 1e-4
        
    @register_pytree_node_class
    class StochasticRNNCell(nn.Module):
        def setup(self):
            self.output_dim = output_dim
            self.rnn_cell = rnn_cell
            self.trunk_fn = trunk_fn
            self.head_mean_fn = head_mean_fn
            self.head_cov_fn = head_cov_fn
            self.eps = eps
        
        def tree_flatten(self):
            children = tuple()
            aux_data = None
            return children, aux_data

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

        def __call__(self, h, u):
            return self._call_single(h, u)

        def _call_single(self, h, u):
            # TODO: figure out the specifics of the architecture...!
            _h = self.trunk_fn(self.rnn_cell(h, u)[0])
            mean, cov_flat = self.head_mean_fn(_h), self.head_cov_fn(_h)
            cov = lie_params_to_constrained(cov_flat, self.output_dim, self.eps)
            return (cov, mean)

    return StochasticRNNCell()