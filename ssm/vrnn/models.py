"""
Implementations of various VRNN models.
"""
import inspect
import jax
import flax
import jax.numpy as np
import jax.random as jr
import flax.linen as nn
import tensorflow_probability.substrates.jax as tfp
from ssm.distributions import MultivariateNormalBlockTridiag
from jax.tree_util import register_pytree_node_class

from ssm import nn_util
from ssm.base import SSM
from ssm.utils import Verbosity, random_rotation, make_named_tuple, ensure_has_batch_dim, auto_batch
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts

tfd = tfp.distributions
SVMPosterior = MultivariateNormalBlockTridiag


@register_pytree_node_class
class VRNN(SSM):
    """
    Follows the definition and nomenclature in Chung et al 2016 [https://arxiv.org/pdf/1506.02216.pdf].

    State is implemented as a tuple.  This will no doubt cause chaos.

    Need to get obs[t] into covariates somehow.

    NOTE - I think the hidden state is shifted by one time index here, but this is waaay easier to implement...
    """

    def __init__(self,
                 num_latent_dims: int,
                 num_emission_dims: int,
                 latent_enc_dim: int,
                 obs_enc_dim: int,
                 rnn_state_dim: int,

                 output_type: str,

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

        self.latent_dim = num_latent_dims
        self.emission_dims = num_emission_dims
        self.latent_enc_dim = latent_enc_dim
        self.obs_enc_dim = obs_enc_dim
        self.rnn_state_dim = rnn_state_dim

        self.output_type = output_type

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
        NOTE - the hidden state part of the initialization is deterministic.  This is normally okay because it is initialized to zeros, but
        we need to double check this in case more exotic initializations are required elsewhere.

        Args:
            covariates:     Not used.  Must be None.
            metadata:       Not used.  Must be None.

        Returns:

        """

        assert covariates is None, "Covariates are not provisioned in the initial state."
        assert metadata is None, "Metadata is not provisioned in the initial state."

        # Initialize the state.  Note that this is normally deterministic so just use a dummy key here.
        initial_h_state = self._rnn_obj.initialize_carry(jr.PRNGKey(0), batch_dims=(), size=self.rnn_state_dim)
        initial_h_dists = jax.tree_map(lambda _state: tfd.Independent(tfd.Deterministic(_state), reinterpreted_batch_ndims=1), initial_h_state)
        initial_h_dist = tfd.JointDistributionSequential(initial_h_dists)

        # Construct the initial distribution over the z latent r.v.
        initial_z_dist = tfd.MultivariateNormalDiag(np.zeros(self.latent_dim), np.ones(self.latent_dim))

        # Construct a joint distribution so that we have a single object.
        initial_dist = tfd.JointDistributionSequential((initial_h_dist, initial_z_dist), )

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

        # # Grab the previous state bits.
        prev_rnn_h = state[0]
        prev_rnn_z = state[1]
        prev_obs = covariates[0]

        # Get and decode the previous latent state.
        prev_rnn_z_dec = self._decoder_latent_obj.apply(self._params_decoder_latent, prev_rnn_z)

        # Encode the PREVIOUS observation.
        prev_obs_enc = self._encoder_data_obj.apply(self._params_encoder_data, prev_obs)

        # Iterate the RNN.
        input_rnn = np.concatenate((prev_rnn_z_dec, prev_obs_enc), axis=-1)
        new_rnn_h, new_rnn_h_exposed = self._rnn_obj.apply(self._params_rnn, prev_rnn_h, input_rnn)

        # Call the *prior* local parameter generation functions.
        rnn_z_dist_local_params = self._prior_obj.apply(self._params_prior, new_rnn_h_exposed)
        rnn_z_dist_local_params = np.reshape(rnn_z_dist_local_params, (*rnn_z_dist_local_params.shape[:-1], 2, -1))
        rnn_z_mean = rnn_z_dist_local_params[..., 0, :]
        rnn_z_log_var = rnn_z_dist_local_params[..., 1, :]

        # Construct the latent distribution itself.
        rnn_z_dist = tfd.MultivariateNormalDiag(rnn_z_mean, np.sqrt(np.exp(rnn_z_log_var)))

        # # # Construct a joint distribution so that we have a single object.
        rnn_h_dists = jax.tree_map(lambda _state: tfd.Independent(tfd.Deterministic(_state), reinterpreted_batch_ndims=1), new_rnn_h)
        rnn_h_dist = tfd.JointDistributionSequential(rnn_h_dists)
        rnn_dist = tfd.JointDistributionSequential((rnn_h_dist, rnn_z_dist), )

        return rnn_dist

    def _emissions(self, state, covariates, metadata):
        """
        oooo baby there is some head scratching indexing in here!
        Args:
            state:
            covariates:
            metadata:

        Returns:

        """

        assert metadata is None, "Metadata is not provisioned in the dynamics."

        # Pull apart the previous VRNN state.
        rnn_h = state[0]
        rnn_z = state[1]

        # Decode the latent through the "prior".
        latent_dec = self._decoder_latent_obj.apply(self._params_decoder_latent, rnn_z)

        # Construct the input to the emissions distribution.
        input_full_decoder = np.concatenate((rnn_h[1], latent_dec, ), axis=-1)

        # Construct the emissions distribution itself.
        emissions_local_parameters = self._decoder_full_obj.apply(self._params_decoder_full, input_full_decoder)

        if self.output_type == 'GAUSSIAN':

            emissions_local_parameters = np.reshape(emissions_local_parameters, (*emissions_local_parameters.shape[:-1], 2, -1))
            emissions_mean = emissions_local_parameters[..., 0, :]
            emissions_log_var = emissions_local_parameters[..., 1, :]
            emissions_dist = tfd.MultivariateNormalDiag(emissions_mean, np.sqrt(np.exp(emissions_log_var)))

        elif self.output_type == 'BERNOULLI':

            emissions_odds_logit = emissions_local_parameters
            emissions_dist = tfd.Independent(tfd.Bernoulli(emissions_odds_logit), reinterpreted_batch_ndims=1)

        else:
            raise NotImplementedError

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
                    self.output_type,
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

    def sample(self, key, num_steps, initial_state=None, covariates=None, metadata=None, num_samples=1):
        """

        Args:
            key:
            num_steps:
            initial_state:
            covariates:
            metadata:
            num_samples:

        Returns:

        """
        assert initial_state is None, "[ERROR}: Cannot use custom inital states (yet)."
        state_history, obs_history = jax.vmap(lambda _k: self._single_unconditional_sample(_k, num_steps))(jr.split(key, num_samples))
        return state_history, obs_history

    def unconditional_sample(self, key, num_steps, num_samples):
        """

        Args:
            key:
            num_steps:
            num_samples:

        Returns:

        """
        return self.sample(key, num_steps, num_samples=num_samples)


class VrnnFilteringProposal(proposals.IndependentGaussianProposal):
    r"""
    Filtering proposal as initially described in the VRNN paper.

    Takes as input the encoded observation and the current rnn state:
        $z_t | x_t \sim N ( \mu_t , diag(\sigma_t^2) )$

    where:
        $[\mu_t \sigma_t^2] = \phi_{\tau}^{enc} ( \phi_{\tau}^x (x_t) )$

    Essentially the same as `proposals.IGSingleObsProposal` but where only the first
    element of `particles` is picked off in the input.

    """

    def _proposal_input_generator(self, dataset, model, particles, t, p_dist, q_state, *_):
        """
        NOTE - inputs are not used here.
        """
        assert self.proposal_window_length == 1, "ERROR: Must have a single-length window."

        raw_obs = jax.lax.dynamic_index_in_dim(dataset, t)
        encoded_obs = model._encoder_data_obj.apply(model._params_encoder_data, raw_obs)

        # TODO - this assumes that the VRNN is using an LSTM.
        # Get the exposed state of the VRNN.
        vrnn_carry, _ = particles
        vrnn_exposed_hidden_state = vrnn_carry[1]

        # Construct the inputs.
        proposal_inputs = (encoded_obs, vrnn_exposed_hidden_state)

        # Map the application if required.
        model_latent_shape = (model.latent_dim, )
        is_batched = (model_latent_shape != vrnn_exposed_hidden_state.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)


class VrnnSmoothingProposal(proposals.IndependentGaussianProposal):
    r"""
    Smoothing VRNN proposal.
    """

    def _proposal_input_generator(self, dataset, model, particles, t, p_dist, q_state, *inputs):
        """
        NOTE - Encoded data (from an external RNN) are supplied through `inputs`.

        This proposal gets a single encoded datapoint (using the VRNN encoder), the data encoder state, and the current particles.
        """
        assert self.proposal_window_length is None, "ERROR: Cannot use a window."
        assert self.n_proposals == 1, "Can only use a single proposal."

        raw_obs = jax.lax.dynamic_index_in_dim(dataset, t)
        encoded_obs = model._encoder_data_obj.apply(model._params_encoder_data, raw_obs)

        # TODO - this assumes that the VRNN is using an LSTM.
        # Get the exposed state of the VRNN.
        vrnn_carry, _ = particles
        vrnn_exposed_hidden_state = vrnn_carry[1]

        # Grab the inputs (formatted as ( (forward_message, backwards_message), ).
        encoder_exposed_hidden_state = inputs[0]

        # Construct the inputs.
        proposal_inputs = (encoded_obs, vrnn_exposed_hidden_state, encoder_exposed_hidden_state)

        # Map the application if required.
        model_latent_shape = (model.latent_dim, )
        is_batched = (model_latent_shape != vrnn_exposed_hidden_state.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0, None))
            return vmapped(*proposal_inputs)


class VrnnRawWindowTilt(tilts.IGWindowTilt):
    r"""
    Tilt for use with VRNN.
    """

    def _tilt_input_generator(self, dataset, model, particles, t, *_):
        """

        """
        assert self.n_tilts == 1, "Can only use a single tilt."

        # TODO - this assumes that the VRNN is using an LSTM.
        vrnn_carry, vrnn_latent = particles
        vrnn_exposed_hidden_state = vrnn_carry[1]

        # Build up the tilt inputs.
        tilt_inputs = (vrnn_latent, vrnn_exposed_hidden_state)

        # Test if the inputs are batched and then apply as required.
        model_latent_shape = (model.latent_dim, )
        is_batched = (model_latent_shape != vrnn_exposed_hidden_state.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(tilt_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree)
            return vmapped(*tilt_inputs)


class VrnnSharedEncodedWindowTilt(tilts.IGWindowTilt):
    r"""
    Tilt for use with VRNN.
    """

    def _tilt_input_generator(self, dataset, model, particles, t, *_):
        """

        """
        assert self.n_tilts == 1, "Can only use a single tilt."

        # TODO - this assumes that the VRNN is using an LSTM.
        vrnn_carry, vrnn_latent = particles
        vrnn_exposed_hidden_state = vrnn_carry[1]

        # Build up the tilt inputs.
        tilt_inputs = (vrnn_latent, vrnn_exposed_hidden_state)

        # Test if the inputs are batched and then apply as required.
        model_latent_shape = (model.latent_dim, )
        is_batched = (model_latent_shape != vrnn_exposed_hidden_state.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(tilt_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree)
            return vmapped(*tilt_inputs)

    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, _tilt_window_length, *_):
        encoded =


class VrnnEncodedTilt(tilts.IndependentGaussianTilt):
    r"""
    Tilt for use with VRNN.
    """

    def _tilt_input_generator(self, dataset, model, particles, t, *inputs):
        """

        """
        assert self.n_tilts == 1, "Can only use a single tilt."
        assert self.tilt_window_length is None, "Cannot use a window here."

        # TODO - this assumes that the VRNN is using an LSTM.
        vrnn_carry, vrnn_latent = particles
        vrnn_exposed_hidden_state = vrnn_carry[1]

        # Build up the tilt inputs.
        tilt_inputs = (vrnn_latent, vrnn_exposed_hidden_state)

        # Test if the inputs are batched and then apply as required.
        model_latent_shape = (model.latent_dim,)
        is_batched = (model_latent_shape != vrnn_exposed_hidden_state.shape)
        if not is_batched:
            return nn_util.vectorize_pytree(tilt_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree)
            return vmapped(*tilt_inputs)

    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, tilt_window_length, *inputs):
        """
        NOTE - external inputs are supplied through the `inputs` variable (as ((forward, backward), ).
        """
        assert tilt_window_length is None, "Cannot use a window here."
        # assert t < len(dataset), "Cannot tilt at the last timestep."

        encoded_future_obs = inputs[0][1]
        encoded_future_obs_at_tp1 = jax.lax.dynamic_index_in_dim(encoded_future_obs, t+1, axis=0, keepdims=False)

        tilt_outputs = (encoded_future_obs_at_tp1, )

        return nn_util.vectorize_pytree(tilt_outputs)
