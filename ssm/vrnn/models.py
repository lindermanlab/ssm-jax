"""
Implementations of various VRNN models.
"""
import inspect
import jax
import flax

import jax.numpy as np
import jax.random as jr
import flax.linen as nn

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

import ssm.inference.proposals as proposals

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.distributions import MultivariateNormalBlockTridiag
SVMPosterior = MultivariateNormalBlockTridiag


@register_pytree_node_class
class VRNN(SSM):
    """
    Follows the definition and nomenclature in Chung et al 2016 [https://arxiv.org/pdf/1506.02216.pdf].

    State is implemented as a tuple.  This will no doubt cause chaos.

    Need to get obs[t] into covariates somehow.

    NOTE - TODO - i think the hidden state is shifted by one time index here, but this is waaay easier to implement...
    """

    def __init__(self,
                 num_latent_dims: int,
                 num_emission_dims: int,
                 latent_enc_dim: int,
                 obs_enc_dim: int,
                 rnn_state_dim: int,

                 rebuild_rnn,                                       # Static functions.
                 rebuild_prior,                                     # Static functions.
                 rebuild_decoder_latent,                            # Static functions.
                 rebuild_decoder_full,                              # Static functions.
                 rebuild_encoder_data,                              # Static functions.

                 params_rnn: flax.core.FrozenDict,                  # Frozen dicts of parameters.
                 params_prior: flax.core.FrozenDict,                # Frozen dicts of parameters.
                 params_decoder_latent: flax.core.FrozenDict,       # Frozen dicts of parameters.
                 params_decoder_full: flax.core.FrozenDict,         # Frozen dicts of parameters.
                 params_encoder_data: flax.core.FrozenDict,         # Frozen dicts of parameters.

                 seed: jr.PRNGKey = None):
        """

        Args:
            num_latent_dims:
            num_emission_dims:
            latent_enc_dim:
            obs_enc_dim:
            rnn_state_dim:

            params_rnn:
            params_prior:
            params_decoder_latent:
            params_decoder_full:
            params_encoder_data:

            rebuild_rnn:
            rebuild_prior:
            rebuild_decoder_latent:
            rebuild_decoder_full:
            rebuild_encoder_data:

            seed:
        """

        # We are only considering the univariate case.
        self.latent_dim = num_latent_dims
        self.emission_dims = num_emission_dims
        self.latent_enc_dim = latent_enc_dim
        self.obs_enc_dim = obs_enc_dim
        self.rnn_state_dim = rnn_state_dim

        self._params_rnn = params_rnn
        self._params_prior = params_prior
        self._params_decoder_latent = params_decoder_latent
        self._params_decoder_full = params_decoder_full
        self._params_encoder_data = params_encoder_data

        self._rebuild_rnn_obj = rebuild_rnn                                     # f_{\theta} ( encoder_data_obj , decoder_latent_obj , h_{t-1} ).
        self._rebuild_prior_obj = rebuild_prior                                 # \phi_{\tau}^{prior} ( h_{t-1} ).
        self._rebuild_decoder_latent_obj = rebuild_decoder_latent               # \phi_{\tau}^{z} ( z_t ).
        self._rebuild_decoder_full_obj = rebuild_decoder_full                   # \phi_{\tau}^{dec} ( \phi_{\tau}^{z} ( z_t ) , h_{t-1} ).
        self._rebuild_encoder_data_obj = rebuild_encoder_data                   # \phi_{\tau}^{x} ( x_{t} ).

        self._rnn_obj = self._rebuild_rnn_obj(params_rnn)                                      # f_{\theta} ( encoder_data_obj , decoder_latent_obj , h_{t-1} ).
        self._prior_obj = self._rebuild_prior_obj(params_prior)                                # \phi_{\tau}^{prior} ( h_{t-1} ).
        self._decoder_latent_obj = self._rebuild_decoder_latent_obj(params_decoder_latent)     # \phi_{\tau}^{z} ( z_t ).
        self._decoder_full_obj = self._rebuild_decoder_full_obj(params_decoder_full)           # \phi_{\tau}^{dec} ( \phi_{\tau}^{z} ( z_t ) , h_{t-1} ).
        self._encoder_data_obj = self._rebuild_encoder_data_obj(params_encoder_data)           # \phi_{\tau}^{x} ( x_{t} ).

        # Grab the parameter values.  This allows us to explicitly re-build the object.
        self._parameters = make_named_tuple(dict_in=locals(),
                                            keys=list(inspect.signature(self.__init__)._parameters.keys()),
                                            name=str(self.__class__.__name__) + 'Tuple')

    def _initial_condition(self, covariates=None, metadata=None):
        """

        Args:
            covariates:     Not used.  Must be None.
            metadata:       Not used.  Must be None.

        Returns:

        """

        assert covariates is None, "Covariates are not provisioned in the initial state."
        assert metadata is None, "Metadata is not provisioned in the initial state."

        initial_rnn_state = np.zeros(self.rnn_state_dim)
        initial_z_state = np.zeros(self.latent_dim)

        # Construct a joint distribution so that we have a single object.
        initial_dist = tfd.JointDistributionSequential((tfd.Deterministic(initial_rnn_state),
                                                        tfd.Deterministic(initial_z_state)), )

        return initial_dist

    def _dynamics(self, state, covariates, metadata=None):
        r"""

        Args:
            state:          PREVIOUS total RNN state `( h_{t-1}, z_{t-1} )`.
            covariates:     Observation: `x_{t}`.
            metadata:       Not used.  Must be None.

        Returns:            DISTRIBUTION over PREVIOUS total RNN state `p( h_{t-1}, z_{t-1} )`.

        """

        assert metadata is None, "Metadata is not provisioned in the dynamics."

        # Grab the previous rnn state.
        prev_rnn_state = state[0]

        # Get and decode the previous latent state.
        prev_latent = state[1]
        prev_latent_dec = self._decoder_latent_obj(prev_latent)

        # Encode the PREVIOUS observation.
        prev_obs = covariates[0]
        prev_obs_enc = self._encoder_data_obj(prev_obs)

        # Iterate the RNN.
        input_rnn = np.concatenate((prev_rnn_state, prev_latent_dec, prev_obs_enc))
        new_rnn_state = self._rnn_obj(input_rnn)
        # new_rnn_state_dist = tfd.Deterministic(new_rnn_state)
        new_rnn_state_dist = tfd.Independent(tfd.Deterministic(new_rnn_state), reinterpreted_batch_ndims=1)  # Was giving a [] event dim.

        # Call the prior local parameter generation functions.
        prior_dist_local_params = self._prior_obj(new_rnn_state)
        prior_dist_local_params = np.reshape(prior_dist_local_params, (2, -1))
        prior_mean = prior_dist_local_params[0]
        prior_log_var = prior_dist_local_params[1]

        # Construct the latent distribution itself.
        prior_latent_dist = tfd.MultivariateNormalDiag(prior_mean, np.sqrt(np.exp(prior_log_var)))

        # Construct a joint distribution so that we have a single object.
        joint_dist = tfd.JointDistributionSequential((new_rnn_state_dist, prior_latent_dist))

        return joint_dist

    def _emissions(self, state, covariates, metadata):
        """

        Args:
            state:
            covariates:
            metadata:

        Returns:

        """

        assert metadata is None, "Metadata is not provisioned in the dynamics."

        # Pull apart the previous state.
        rnn_state = state[0]
        latent = state[1]
        latent_dec = self._decoder_latent_obj(latent)

        # Construct the input to the emissions distribution.
        input_full_decoder = np.concatenate((rnn_state, latent_dec, ))

        # Construct the emissions distribution itself.
        emissions_local_parameters = self._decoder_full_obj(input_full_decoder)
        emissions_local_parameters = np.reshape(emissions_local_parameters, (2, -1))
        emissions_mean = emissions_local_parameters[0]
        emissions_log_var = emissions_local_parameters[1]
        emissions_dist = tfd.MultivariateNormalDiag(emissions_mean, np.sqrt(np.exp(emissions_log_var)))

        return emissions_dist

    def tree_flatten(self):
        children = (self._params_rnn,
                    self._params_prior,
                    self._params_decoder_latent,
                    self._params_decoder_full,
                    self._params_encoder_data,
                    )
        aux_data = (self.latent_dim,
                    self.emission_dims,
                    self.latent_enc_dim,
                    self.obs_enc_dim,
                    self.rnn_state_dim,
                    self._rebuild_rnn_obj,
                    self._rebuild_prior_obj,
                    self._rebuild_decoder_latent_obj,
                    self._rebuild_decoder_full_obj,
                    self._rebuild_encoder_data_obj,
                    )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

    @property
    def emissions_shape(self):
        return self.emission_dims,

    def initial_distribution(self,
                             covariates=None,
                             metadata=None):
        return self._initial_condition(covariates, metadata)

    def dynamics_distribution(self,
                              state,
                              covariates=None,
                              metadata=None):
        # assert covariates is not None, 'Inputs must come in through `covariates`.'
        return self._dynamics(state, covariates, metadata)

    def emissions_distribution(self,
                               state,
                               covariates=None,
                               metadata=None):
        return self._emissions(state, covariates, metadata)

    @ensure_has_batch_dim()
    def m_step(self,
               data: np.ndarray,
               posterior: SVMPosterior,
               covariates=None,
               metadata=None,
               key: jr.PRNGKey=None):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _single_unconditional_sample(self, key, num_steps):
        """

        Args:
            key:
            numsteps:

        Returns:

        """
        key, subkey = jr.split(key)
        initial_particles = self.initial_distribution().sample(seed=key)

        key, subkey = jr.split(key)
        initial_obs = self.emissions_distribution(initial_particles).sample(seed=key)

        state_history = [initial_particles]
        obs_history = [initial_obs]

        for _t in range(1, num_steps):
            # Iterate the model
            key, subkey = jr.split(key)
            new_state = self.dynamics_distribution(state_history[-1], covariates=(obs_history[-1], )).sample(seed=subkey)

            # Generate the emission.
            key, subkey = jr.split(key)
            new_obs = self.emissions_distribution(new_state, ).sample(seed=subkey)

            # Append to the history.
            state_history.append(new_state)
            obs_history.append(new_obs)

        state_history = (np.stack([_s[0] for _s in state_history]),
                         np.stack([_s[1] for _s in state_history]))
        obs_history = np.asarray(obs_history)

        return state_history, obs_history

    def unconditional_sample(self, key, num_steps, num_samples):
        """

        Args:
            key:
            num_steps:
            num_samples:

        Returns:

        """

        state_history, obs_history = jax.vmap(lambda _k: self._single_unconditional_sample(_k, num_steps))(jr.split(key, num_samples))

        return state_history, obs_history


# def define_vrnn_proposal(_key,
#                          _dataset,
#                          _vrnn: VRNN):
#
#     trunk_fn = None
#     head_mean_fn = nn.Dense(val_latent)
#     head_log_var_fn = nn.Dense(val_latent)
#
#     particles = val_rnn_state
#     q_state = val_data_encoded
#
#     # Format: (dataset, model, particles, t, p_dist, q_state)
#     dummy_particles = np.repeat(np.expand_dims(val_rnn_state, 0), 2)
#     dummy_p_dist = _vrnn.dynamics_distribution(dummy_particles, )
#     stock_proposal_input = (_dataset, _vrnn, dummy_particles, 0, dummy_p_dist)
#
#     return proposals.IndependentGaussianProposal(1, )


# def define_vrnn_model(_key):
#
#     kernel_init = lambda *args: nn.initializers.lecun_normal()(*args) * 0.0001
#
#     rnn = nn.Dense(rnn_state_dim, kernel_init=kernel_init)
#     prior = nn.Dense(2 * latent_dim, kernel_init=kernel_init)
#     latent_decoder = nn.Dense(latent_encoded_dim, kernel_init=kernel_init)
#     full_decoder = nn.Dense(2 * emissions_dim, kernel_init=kernel_init)
#     data_encoder = nn.Dense(emissions_encoded_dim, kernel_init=kernel_init)
#
#     input_prior = val_rnn_state
#     input_rnn = np.concatenate((val_rnn_state, val_latent_decoded, val_data_encoded))
#     input_latent_decoder = val_latent
#     input_full_decoder = np.concatenate((val_rnn_state, val_latent_decoded))
#     input_data_encoder = val_obs
#
#     _key, *subkeys = jr.split(_key, num=6)
#     params_rnn = rnn.init(subkeys[0], input_rnn)
#     params_prior = prior.init(subkeys[1], input_prior)
#     params_latent_decoder = latent_decoder.init(subkeys[2], input_latent_decoder)
#     params_full_decoder = full_decoder.init(subkeys[3], input_full_decoder)
#     params_data_encoder = data_encoder.init(subkeys[4], input_data_encoder)
#
#     rebuild_rnn = lambda _params: rnn.bind(_params)
#     rebuild_prior = lambda _params: prior.bind(_params)
#     rebuild_latent_decoder = lambda _params: latent_decoder.bind(_params)
#     rebuild_full_decoder = lambda _params: full_decoder.bind(_params)
#     rebuild_data_encoder = lambda _params: data_encoder.bind(_params)
#
#     return VRNN(latent_dim,
#                 emissions_dim,
#                 latent_encoded_dim,
#                 emissions_encoded_dim,
#                 rnn_state_dim,
#                 rebuild_rnn,
#                 rebuild_prior,
#                 rebuild_latent_decoder,
#                 rebuild_full_decoder,
#                 rebuild_data_encoder,
#                 params_rnn,
#                 params_prior,
#                 params_latent_decoder,
#                 params_full_decoder,
#                 params_data_encoder, )


if __name__ == '__main__':

    rnn_state_dim = 10
    latent_dim = 11
    emissions_dim = 12
    latent_encoded_dim = 5
    emissions_encoded_dim = 6

    val_rnn_state = np.zeros(rnn_state_dim)
    val_obs = np.zeros(emissions_dim)
    val_latent = np.zeros(latent_dim)
    val_latent_decoded = np.zeros(latent_encoded_dim)
    val_data_encoded = np.zeros(emissions_encoded_dim)

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    vrnn = define_vrnn_model(subkey)

    # Do some jit tests.
    def fn(_key, _vrnn):
        return _vrnn.emissions_shape

    jitted = jax.jit(fn)

    a = jitted(jr.PRNGKey(0), vrnn)
    b = jitted(jr.PRNGKey(1), vrnn)
    c = jax.vmap(jitted, in_axes=(0, None))(jr.split(jr.PRNGKey(0), 10), vrnn, )

    # Test some iteration stuff.
    key, subkey = jr.split(key)
    initial_particles = vrnn.initial_distribution().sample(seed=subkey)

    # Generate the initial observations.  NOTE - the subroutine requires previous and current obs, so just duplicate.
    key, subkey = jr.split(key)
    initial_obs = (vrnn.emissions_distribution(initial_particles).sample(seed=subkey),
                   vrnn.emissions_distribution(initial_particles).sample(seed=subkey))

    iterated = vrnn.dynamics_distribution(initial_particles, covariates=initial_obs)

    print('Done')



