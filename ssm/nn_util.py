# Stolen from Andy's smc branch

import jax
from jax import vmap
import jax.numpy as np
import flax.linen as nn
from jax.tree_util import register_pytree_node_class
import jax.scipy as scipy

from typing import (NamedTuple, Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inv_softplus(x, eps=1e-4):
    return np.log(np.exp(x - eps) - 1)

def vectorize_pytree(*args):
    """
    Flatten an arbitrary PyTree into a vector.
    :param args:
    :return:
    """
    flat_tree, _ = jax.tree_util.tree_flatten(args)
    flat_vs = [x.flatten() for x in flat_tree]
    return np.concatenate(flat_vs, axis=0)

# converts an (n(n+1)/2,) vector of Lie parameters
# to an (n, n) matrix
def lie_params_to_constrained(out_flat, dim, eps=1e-4):
    D, A = out_flat[:dim], out_flat[dim:]
    # ATTENTION: we changed this!
    # D = np.maximum(softplus(D), eps)
    D = softplus(D) + eps
    # Build a skew-symmetric matrix
    S = np.zeros((dim, dim))
    i1, i2 = np.tril_indices(dim - 1)
    S = S.at[i1+1, i2].set(A)
    S = S.T
    S = S.at[i1+1, i2].set(-A)

    O = scipy.linalg.expm(S)
    J = O.T @ np.diag(D) @ O
    return J

class MLP(nn.Module):
    """
    Define a simple fully connected MLP with ReLU activations.
    """
    features: Sequence[int]
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init, )(x))
        x = nn.Dense(self.features[-1], 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init)(x)
        return x

class Identity(nn.Module):
    """
    A layer which passes the input through unchanged.
    """
    features: int

    def __call__(self, inputs):
        return inputs

class Static(nn.Module):
    """
    A layer which just returns some static parameters.
    """
    features: int
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init,
                            (self.features, ))
        return kernel

class PotentialNetwork(nn.Module):
    def __call__(self, inputs):
        """
        Equivalent to the `.forward` method in PyTorch.  Generates the parameters of an independent
        multivariate Gaussian distribution as a function of the inputs.
        Args:
            inputs:
        Returns:
        """
        J_diag, h = self._generate_distribution_parameters(inputs)
        return (J_diag, h)

    def _generate_distribution_parameters(self, inputs):
        """
        Map over the inputs if there are multiple inputs, otherwise, apply it to a single input.
        Whether there is a batch dimension is established by whether the whole shape of the input equals the
        pre-specified input shape.  If they are unequal, assumes that the first dimension is a batch dimension.
        Args:
            inputs (ndarray):   Possibly batched set of inputs (if batched, first dimension is batch dimension).
        Returns:
            (ndarray):          Possibly batched set of Gaussian parameters (if batched, first dimension is batch
                                dimension).
        """
        if (len(inputs.shape) == 3):
            # We have both a batch dimension and a time dimension
            # and we have to vmap over both...!
            return vmap(vmap(self._call_single, 0), 0)(inputs)
        elif (len(inputs.shape) == 2):
            return vmap(self._call_single)(inputs)
        else:
            return self._call_single(inputs)

    def _call_single(self, inputs):
        """
        Args:
            inputs (ndarray):   Single input data point (NO batch dimension).
        Returns:
            (ndarray):          Single mean and variance VECTORS representing the parameters of a single
                                independent multivariate normal distribution.
        """
        pass

class GaussianNetworkDiag(PotentialNetwork):

    latent_dim : int = None
    trunk_fn : nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, output_dim, input_dim = None,
                    trunk_fn=None, head_mean_fn=None, head_log_var_fn=None,
                    eps = 1e-4,
                    **kwargs): 
        network_params = kwargs["recnet_architecture"]
        output_dim = network_params.get("latent_dim") or output_dim
        
        out_trunk_features = network_params.get("out_trunk_features")
        if out_trunk_features == [] or out_trunk_features is None:
            trunk_fn = trunk_fn or Identity(input_dim)
        else:
            trunk_fn = trunk_fn or MLP(out_trunk_features)

        head_mean_fn = head_mean_fn or MLP(network_params["out_mean_features"] 
                                           + [output_dim,])
        head_log_var_fn = head_log_var_fn or MLP(network_params["out_var_features"] 
                                                 + [output_dim,])
        
        if (network_params.get("static_covariance")):
            head_log_var_fn = Static(output_dim, nn.initializers.constant(1))

        eps = network_params.get("eps") or eps

        return cls(output_dim, trunk_fn, head_mean_fn, head_log_var_fn, eps)

    def tree_flatten(self):
        children = (self.latent_dim,
                    self.trunk_fn,
                    self.head_mean_fn,
                    self.head_log_var_fn,
                    self.eps)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def _call_single(self, inputs):
        # Flatten the input.
        inputs_flat = vectorize_pytree(inputs)
        # Apply the trunk.
        trunk_output = self.trunk_fn(inputs_flat)
        # Get the mean.
        h_flat = self.head_mean_fn(trunk_output)
        h = h_flat
        # Get the variance output and reshape it.
        var_output_flat = self.head_log_var_fn(trunk_output)
        J_diag = np.maximum(softplus(var_output_flat), self.eps)
        J_diag = np.diag(J_diag)
        return (J_diag, h)

class GaussianNetworkFull(PotentialNetwork):

    latent_dim : int = None
    trunk_fn : nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, output_dim, input_dim = None,
                    trunk_fn=None, head_mean_fn=None, head_log_var_fn=None,
                    eps=1e-4, **kwargs): 

        network_params = kwargs["recnet_architecture"]
        out_trunk_features = network_params.get("out_trunk_features")
        if out_trunk_features == [] or out_trunk_features is None:
            trunk_fn = trunk_fn or Identity(input_dim)
        else:
            trunk_fn = trunk_fn or MLP(out_trunk_features)
            
        head_mean_fn = head_mean_fn or MLP(
            network_params["out_mean_features"] + [output_dim,])

        init_cov_bias = network_params["cov_init"]

        head_log_var_fn = head_log_var_fn or MLP(
            network_params["out_var_features"] + [output_dim * (output_dim + 1) // 2,], 
            nn.initializers.zeros, 
            nn.initializers.constant(init_cov_bias))

        if (network_params.get("static_covariance")):
            head_log_var_fn = Static(output_dim * (output_dim + 1) // 2, 
                                     nn.initializers.constant(init_cov_bias))
        
        eps = network_params.get("eps") or eps

        return cls(output_dim, trunk_fn, head_mean_fn, head_log_var_fn, eps)
    
    def _call_single(self, inputs):
        # Flatten the input.
        inputs_flat = vectorize_pytree(inputs)
        # Apply the trunk.
        trunk_output = self.trunk_fn(inputs_flat)
        # Get the mean.
        h_flat = self.head_mean_fn(trunk_output)
        h = h_flat
        # Get the covariance parameters and build a full matrix from it.
        var_output_flat = self.head_log_var_fn(trunk_output)
        J = lie_params_to_constrained(var_output_flat, self.latent_dim, self.eps)
        return (J, h)

class GaussianNetworkFullMeanParams(GaussianNetworkFull):

    latent_dim : int = None
    trunk_fn : nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    eps : float = None
    
    def _call_single(self, inputs):
        # Flatten the input.
        inputs_flat = vectorize_pytree(inputs)
        # Apply the trunk.
        trunk_output = self.trunk_fn(inputs_flat)
        # Get the mean.
        mu = self.head_mean_fn(trunk_output)
        # Get the covariance parameters and build a full matrix from it.
        var_output_flat = self.head_log_var_fn(trunk_output)
        Sigma = lie_params_to_constrained(var_output_flat, self.latent_dim, self.eps)
        h = np.linalg.solve(Sigma, mu)
        J = np.linalg.inv(Sigma)
        # pdb.set_trace()
        return (J, h)

# Outputs Gaussian distributions for the entire sequence at once
class PosteriorNetwork(PotentialNetwork):
    def __call__(self, inputs):
        J_diag, L_diag, h = self._generate_distribution_parameters(inputs)
        return (J_diag, L_diag, h)

    def _generate_distribution_parameters(self, inputs):
        """
        Map over the inputs if there are multiple inputs, otherwise, apply it to a single input.
        Whether there is a batch dimension is established by whether the whole shape of the input equals the
        pre-specified input shape.  If they are unequal, assumes that the first dimension is a batch dimension.
        Args:
            inputs (ndarray):   Possibly batched set of inputs (if batched, first dimension is batch dimension).
        Returns:
            (ndarray):          Possibly batched set of Gaussian parameters (if batched, first dimension is batch
                                dimension).
        """
        is_batched = (len(inputs.shape) == 3)
        if is_batched:
            return vmap(self._call_single, in_axes=0)(inputs)
        else:
            assert(len(inputs.shape) == 2)
            return self._call_single(inputs)

class BiRNN(PosteriorNetwork):
    
    latent_dim : int = None
    output_dim : int = None
    forward_RNN : nn.Module = None
    backward_RNN : nn.Module = None
    input_fn : nn.Module = None
    trunk_fn: nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, output_dim, input_dim=None, cell_type=nn.GRUCell,
                    forward_RNN=None, backward_RNN=None, trunk_fn=None,
                    input_fn=None, head_mean_fn=None, head_log_var_fn=None,
                    eps=1e-4, **kwargs): 

        forward_RNN = forward_RNN or nn.scan(cell_type, variable_broadcast="params", 
                                             split_rngs={"params": False})()
        backward_RNN = backward_RNN or nn.scan(cell_type, variable_broadcast="params", 
                                               split_rngs={"params": False}, reverse=True)()
        
        rnn_params = kwargs["recnet_architecture_rnn"]

        out_trunk_features = rnn_params.get("out_trunk_features")
        if out_trunk_features == [] or out_trunk_features is None:
            trunk_fn = trunk_fn or Identity(input_dim)
        else:
            trunk_fn = trunk_fn or MLP(out_trunk_features)

        head_mean_fn = head_mean_fn or MLP(rnn_params["out_mean_features"] + [output_dim,])
        init_cov_bias = rnn_params["cov_init"]

        head_log_var_fn = head_log_var_fn or MLP(
            rnn_params["out_var_features"] + [output_dim * (output_dim + 1) // 2,], 
            nn.initializers.zeros, 
            nn.initializers.constant(init_cov_bias))
        
        latent_dim = rnn_params["latent_dim"] or output_dim
        input_fn = input_fn or MLP(rnn_params["in_features"] + [latent_dim,])

        eps = rnn_params.get("eps") or eps

        return cls(latent_dim, output_dim, forward_RNN, backward_RNN, 
                   input_fn, trunk_fn, head_mean_fn, head_log_var_fn, eps)

    # Applied the BiRNN to a single sequence of inputs
    def _call_single(self, inputs):
        
        inputs = vmap(self.input_fn)(inputs)

        init_carry_forward = np.zeros((self.latent_dim,))
        _, out_forward = self.forward_RNN(init_carry_forward, inputs)
        
        init_carry_backward = np.zeros((self.latent_dim,))
        _, out_backward = self.backward_RNN(init_carry_backward, inputs)

        # Concatenate the forward and backward outputs
        out_combined = np.concatenate([out_forward, out_backward], axis=-1)
        
        # Get the mean.
        # vmap over the time dimension
        h_flat = vmap(self.head_mean_fn)(out_combined)
        h = h_flat
        # Get the variance output and reshape it.
        # vmap over the time dimension
        var_output_flat = vmap(self.head_log_var_fn)(out_combined)
        J = vmap(lie_params_to_constrained, in_axes=(0, None, None))\
            (var_output_flat, self.output_dim, self.eps)
        
        seq_len, latent_dim, _ = J.shape
        # lower diagonal blocks of precision matrix
        L = np.zeros((seq_len-1, latent_dim, latent_dim))

        return (J, L, h)

class BiRNNMeanParams(BiRNN):
    def _call_single(self, inputs):
        
        inputs = vmap(self.input_fn)(inputs)

        init_carry_forward = np.zeros((self.latent_dim,))
        _, out_forward = self.forward_RNN(init_carry_forward, inputs)
        
        init_carry_backward = np.zeros((self.latent_dim,))
        _, out_backward = self.backward_RNN(init_carry_backward, inputs)

        # Concatenate the forward and backward outputs
        out_combined = np.concatenate([out_forward, out_backward], axis=-1)
        
        # Get the mean.
        # vmap over the time dimension
        mu = vmap(self.head_mean_fn)(out_combined)
        # Get the variance output and reshape it.
        # vmap over the time dimension
        var_output_flat = vmap(self.head_log_var_fn)(out_combined)
        Sigma = vmap(lie_params_to_constrained, in_axes=(0, None, None))\
            (var_output_flat, self.output_dim, self.eps)

        h = vmap(np.linalg.solve, in_axes=(0, 0))(Sigma, mu)
        J = np.linalg.inv(Sigma)

        seq_len, latent_dim, _ = J.shape
        # lower diagonal blocks of precision matrix
        L = np.zeros((seq_len-1, latent_dim, latent_dim))

        return (J, L, h)

# Also uses Lie parameterization
class CBiRNN(PosteriorNetwork):

    latent_dim : int = None
    output_dim : int = None
    forward_RNN : nn.Module = None
    backward_RNN : nn.Module = None
    input_fn : nn.Module = None
    trunk_fn: nn.Module = None
    head_mean_fn : nn.Module = None
    head_log_var_fn : nn.Module = None
    dynamics_fn : nn.Module = None
    eps : float = None

    @classmethod
    def from_params(cls, output_dim, input_dim=None, cell_type=nn.GRUCell,
                    forward_RNN=None, backward_RNN=None, trunk_fn=None,
                    input_fn=None, head_mean_fn=None, head_log_var_fn=None,
                    dynamics_fn=None, eps=1e-4,
                    **kwargs): 

        forward_RNN = forward_RNN or nn.scan(cell_type, variable_broadcast="params", 
                                             split_rngs={"params": False})()
        backward_RNN = backward_RNN or nn.scan(cell_type, variable_broadcast="params", 
                                               split_rngs={"params": False}, reverse=True)()
        
        rnn_params = kwargs["recnet_architecture_rnn"]

        out_trunk_features = rnn_params.get("out_trunk_features")
        if out_trunk_features == [] or out_trunk_features is None:
            trunk_fn = trunk_fn or Identity(input_dim)
        else:
            trunk_fn = trunk_fn or MLP(out_trunk_features)

        head_mean_fn = head_mean_fn or MLP(
            rnn_params["out_mean_features"] + [output_dim,])
        # The full covariance function has D*(D+1)/2 params
        init_cov_bias = rnn_params["cov_init"]
        head_log_var_fn = head_log_var_fn or MLP(rnn_params["out_var_features"] \
                                + [output_dim * (output_dim + 1) // 2,],
                                nn.initializers.zeros, nn.initializers.constant(init_cov_bias))
        dynamics_fn = dynamics_fn or MLP(rnn_params["dyn_features"] + [output_dim ** 2,],
                                nn.initializers.zeros, nn.initializers.zeros)
        latent_dim = rnn_params["latent_dim"] or output_dim
        input_fn = input_fn or MLP(rnn_params["in_features"] + [latent_dim,])

        eps = rnn_params.get("eps") or eps

        return cls(latent_dim, output_dim, forward_RNN, backward_RNN, 
                   input_fn, trunk_fn, head_mean_fn, head_log_var_fn, dynamics_fn, eps)
    
    # Applied the BiRNN to a single sequence of inputs
    def _call_single(self, inputs):
        inputs = vmap(self.input_fn)(inputs)

        init_carry_forward = np.zeros((self.latent_dim,))
        _, out_forward = self.forward_RNN(init_carry_forward, inputs)
        
        init_carry_backward = np.zeros((self.latent_dim,))
        _, out_backward = self.backward_RNN(init_carry_backward, inputs)

        # Concatenate the forward and backward outputs
        out_combined = np.concatenate([out_forward, out_backward], axis=-1)
        
        # Get the mean.
        # vmap over the time dimension
        b = vmap(self.head_mean_fn)(out_combined)

        # Get the variance output and reshape it.
        # vmap over the time dimension
        var_output_flat = vmap(self.head_log_var_fn)(out_combined)

        Q_inv = vmap(lie_params_to_constrained, in_axes=(0, None, None))\
            (var_output_flat, self.output_dim, self.eps)

        dynamics_flat = vmap(self.dynamics_fn)(out_combined)
        A = dynamics_flat.reshape((-1, self.output_dim, self.output_dim))

        L_diag = np.einsum("til,tlj->tij", -Q_inv[1:], A[1:])
        ATQinvA = np.einsum("tji,tjl,tlk->tik", A[1:], Q_inv[1:], A[1:])
        ATQinvb = np.einsum("tli,tlj,tj->ti", A[1:], Q_inv[1:], b[1:])
        # Here the J matrices are full matrices
        J_diag = Q_inv.at[:-1].add(ATQinvA)

        h = np.einsum("tij,tj->ti", Q_inv, b).at[:-1].add(ATQinvb)

        return (J_diag, L_diag, h)