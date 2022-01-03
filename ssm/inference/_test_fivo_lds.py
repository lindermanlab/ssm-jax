import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
import flax.linen as nn
from typing import NamedTuple
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.lds.models import GaussianLDS
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts


def lds_define_test(key, free_parameters, proposal_structure, tilt_structure):
    """

    Args:
        key:
        free_parameters:
        proposal_structure:
        tilt_structure:

    Returns:

    """

    # Define the true model.
    key, subkey = jr.split(key)
    true_model, true_states, dataset = lds_define_true_model_and_data(subkey)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = lds_define_test_model(subkey, true_model, free_parameters)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = lds_define_proposal(subkey, model, dataset, proposal_structure)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = lds_define_tilt(subkey, model, dataset, tilt_structure)

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, dataset)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    return ret_model, ret_test, ret_prop, ret_tilt


def lds_define_test_model(key, true_model, free_parameters):
    """

    Args:
        key:
        true_model:
        free_parameters:

    Returns:

    """
    key, subkey = jr.split(key)

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, free_parameters)

    if len(free_parameters) > 0:

        # Get the default parameters from the true model.
        true_params = fivo.get_model_params_fn(true_model)

        # Generate a model to use.  NOTE - this will generate a new model, and we will
        # overwrite any of the free parameters of interest into the true model.
        tmp_model = true_model.__class__(num_latent_dims=true_model.latent_dim,
                                         num_emission_dims=true_model.emissions_shape[0],
                                         seed=subkey)

        # Dig out the free parameters.
        init_free_params = get_free_model_params_fn(tmp_model)

        # Overwrite all the params with the new values.
        default_params = utils.mutate_named_tuple_by_key(true_params, init_free_params)

        # Mutate the free parameters.
        for _k in free_parameters:
            _base = getattr(default_params, _k)
            key, subkey = jr.split(key)
            new_val = {_k: _base + (10.0 * jr.normal(key=subkey, shape=_base.shape))}
            default_params = utils.mutate_named_tuple_by_key(default_params, new_val)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


class LdsTilt(tilts.IndependentGaussianTilt):
    """

    """

    def apply(self, params, dataset, model, particles, t, *inputs):
        """

        Args:
            params (FrozenDict):    FrozenDict of the parameters of the tilt.

            dataset:

            model:

            particles:

            t:

            inputs (tuple):         Tuple of additional inputs to the tilt in SMC.

            data:

        Returns:
            (Float): Tilt log value.

        """

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = params[0]
        else:
            t_params = jax.tree_map(lambda args: args[t], params)

        # Generate a tilt distribution.
        tilt_inputs = self._tilt_input_generator(dataset, model, particles, t, *inputs)
        r_dist = self.tilt.apply(t_params, tilt_inputs)

        # Now score under that distribution.
        tilt_outputs = self._tilt_output_generator(dataset, model, particles, t, *inputs)

        # There may be NaNs here, so we need to pull this apart.
        means = r_dist.mean().T
        sds = r_dist.variance().T

        # Sweep over the vector and return zeros where appropriate.
        def _eval(_idx, _mu, _sd, _out):
            _dist = tfd.MultivariateNormalDiag(loc=np.expand_dims(_mu, -1), scale_diag=np.sqrt(np.expand_dims(_sd, -1)))
            return jax.lax.cond(_idx < t,
                                lambda *args: np.zeros(means.shape[1]),
                                lambda *args: _dist.log_prob(np.asarray([_out])),
                                None)

        log_r_val = jax.vmap(_eval)(np.arange(means.shape[0]), means, sds, tilt_outputs).sum(axis=0)

        return log_r_val

    # Define a method for generating thei nputs to the tilt.
    def _tilt_input_generator(self, dataset, model, particles, t, *_inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t) into a vector object that
        can be input into the tilt.

        NOTE - because of the conditional independncies introduced by the HMM, there is no dependence
        on the previous states.

        Args:
            dataset:

            model:

            particles:

            t:

            *inputs_:

        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into tilt.

        """

        # Just the particles are passed in.
        tilt_inputs = (particles, )

        is_batched = (model.latent_dim != particles.shape[0])
        if not is_batched:
            return nn_util.vectorize_pytree(tilt_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(0, ))
            return vmapped(*tilt_inputs)

    # We need to define the method for generating the inputs.
    def _tilt_output_generator(self, dataset, model, particles, t, *_inputs):
        """
        Define the output generator for the lds example.
        Args:
            dataset:

            model:

            particles:

            t:

            *inputs_:

        Returns:

        """

        # We will pass in whole data into the tilt and then filter out as required.
        tilt_outputs = (dataset, )

        return nn_util.vectorize_pytree(tilt_outputs)


def lds_define_tilt(subkey, model, dataset, tilt_structure):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """

    if (tilt_structure is None) or (tilt_structure == 'NONE'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Check whether we have a valid number of tilts.
    n_tilts = len(dataset[0]) - 1

    # Tilt functions take in (dataset, model, particles, t-1).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    output_lengths = np.arange(len(dataset[0]), 0, -1)

    stock_tilt_input = (dataset[-1], model, dummy_particles[0], 0)

    dummy_tilt_output = nn_util.vectorize_pytree(dataset[0], )
    head_mean_fn = nn.Dense(dummy_tilt_output.shape[0])
    head_log_var_fn = nn_util.Static(dummy_tilt_output.shape[0])

    # Define the tilts themselves.
    tilt = LdsTilt(n_tilts=n_tilts,
                   tilt_input=stock_tilt_input,
                   head_mean_fn=head_mean_fn,
                   head_log_var_fn=head_log_var_fn)

    tilt_params = tilt.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_tilt_fn = tilts.rebuild_tilt(tilt, tilt_structure)
    return tilt, tilt_params, rebuild_tilt_fn


def lds_define_proposal(subkey, model, dataset, proposal_structure):
    """

    Args:
        subkey:
        model:
        dataset:
        proposal_structure:

    Returns:

    """

    if (proposal_structure is None) or (proposal_structure == 'BOOTSTRAP'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )

    trunk_fn = None
    head_mean_fn = nn.Dense(dummy_proposal_output.shape[0])
    head_log_var_fn = nn_util.Static(dummy_proposal_output.shape[0])

    # trunk_fn = nn_util.MLP([5, 5])
    # head_mean_fn = nn.Dense(dummy_proposal_output.shape[0])
    # head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0])

    # Check whether we have a valid number of proposals.
    n_props = len(dataset[0])

    # Define the required method for building the inputs.
    def lds_proposal_input_generator(_dataset, _model, _particles, _t, _p_dist, _q_state):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t, p_dist, q_state) into a vector object that
        can be input into the proposal.

        Args:


        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into proposal.

        """

        # This proposal gets the entire dataset and the current particles.
        _proposal_inputs = (_dataset, _particles)

        _is_batched = (_model.latent_dim != _particles.shape[0])
        if not _is_batched:
            return nn_util.vectorize_pytree(_proposal_inputs)
        else:
            _vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return _vmapped(*_proposal_inputs)

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=n_props,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output,
                                                     input_generator=lds_proposal_input_generator,
                                                     trunk_fn=trunk_fn,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )

    # Initialize the network.
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def lds_get_true_target_marginal(model, data):
    """
    Take in a model and some data and return the tfd distribution representing the marginals of true posterior.
    Args:
        model:
        data:

    Returns:

    """
    pred_em_posterior = jax.vmap(model.e_step)(data)

    marginal_mean = pred_em_posterior.mean().squeeze()
    marginal_std = np.sqrt(pred_em_posterior.covariance().squeeze())

    pred_em_marginal = tfd.MultivariateNormalDiag(marginal_mean, marginal_std)

    return pred_em_marginal


def lds_define_true_model_and_data(key):
    """

    Args:
        key:

    Returns:

    """
    latent_dim = 2
    emissions_dim = 3
    num_trials = 100000
    T = 9  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.

    # Create a more reasonable emission scale.
    dynamics_scale_tril = 1.0 * np.eye(latent_dim)
    true_dynamics_weights = np.eye(latent_dim)
    true_emission_weights = np.eye(emissions_dim, latent_dim)

    # NOTE - can make observations tighter here.
    # emission_scale_tril = 0.1 * np.eye(emissions_dim)
    emission_scale_tril = 1.0 * np.eye(emissions_dim)

    initial_state_scale_tril = 5.0 * np.eye(latent_dim)

    # Create the true model.
    key, subkey = jr.split(key)

    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=dynamics_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_weights=true_emission_weights,
                             emission_scale_tril=emission_scale_tril,
                             initial_state_scale_tril=initial_state_scale_tril
                             )

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=T+1, num_samples=num_trials)

    return true_model, true_states, dataset


def lds_do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params,
                param_figs):
    """
    NOTE - removed proposal and tilt parameters here as they will be too complex.

    Args:
        _param_hist:
        _loss_hist:
        _true_loss_em:
        _true_loss_smc:
        _true_params:
        param_figs:

    Returns:

    """

    fsize = (12, 8)
    idx_to_str = lambda _idx: ['Model (p): '][_idx]

    for _p, _hist in enumerate(_param_hist[:1]):

        if _hist[0] is not None:
            if len(_hist[0]) > 0:

                n_param = len(_hist[0].keys())

                if param_figs[_p] is None:
                    param_figs[_p] = plt.subplots(n_param, 1, figsize=fsize, sharex=True, squeeze=True)

                for _i, _k in enumerate(_hist[0].keys()):
                    to_plot = []
                    for _pt in _param_hist[_p]:
                        to_plot.append(_pt[_k].flatten())
                    to_plot = np.array(to_plot)

                    if hasattr(param_figs[_p][1], '__len__'):
                        plt.sca(param_figs[_p][1][_i])
                    else:
                        plt.sca(param_figs[_p][1])
                    plt.cla()
                    plt.plot(to_plot)
                    plt.title(idx_to_str(_p) + str(_k))
                    plt.grid(True)
                    plt.tight_layout()
                    plt.pause(0.00001)

                plt.savefig('./lds_param_{}.pdf'.format(_p))

    return param_figs


def lds_do_print(_step, true_model, opt, true_lml, pred_lml, pred_fivo_bound, em_log_marginal_likelihood=None):
    """

    Args:
        _step:
        true_model:
        opt:
        true_lml:
        pred_lml:
        pred_fivo_bound:
        em_log_marginal_likelihood:

    Returns:

    """
    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},  Pred Neg-LML: {: >8.3f},  Pred FIVO bound {: >8.3f}'.\
        format(_step, true_lml, pred_lml, pred_fivo_bound)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    if opt[0] is not None:
        if len(opt[0].target) > 0:
            # print()
            print('\tModel')
            true_bias = true_model.dynamics_bias.flatten()
            pred_bias = opt[0].target[0].flatten()
            print('\t\tTrue: dynamics bias:     ', '  '.join(['{: >9.3f}'.format(_s) for _s in true_bias]))
            print('\t\tPred: dynamics bias:     ', '  '.join(['{: >9.3f}'.format(_s) for _s in pred_bias]))

    # NOTE - the proposal and tilt are more complex here, so don't show them.

    print()
    print()
    print()
