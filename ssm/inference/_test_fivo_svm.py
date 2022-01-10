import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import argparse
from jax import random as jr
import flax.linen as nn
from typing import NamedTuple
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.svm.models import UnivariateSVM
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts


def svm_get_config():
    """

    Returns:

    """

    # Set up the experiment.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SVM', type=str)

    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--log-group', default='debug', type=str)  # {'debug', 'gdm-v1.0'}

    parser.add_argument('--use-sgr', default=1, type=int)  # {0, 1}

    parser.add_argument('--free-parameters', default='invsig_phi', type=str)  # CSV.  # {'log_sigma', 'invsig_phi'}
    parser.add_argument('--proposal-structure', default='DIRECT', type=str)  # {None/'BOOTSTRAP', 'DIRECT', 'RESQ', }
    parser.add_argument('--tilt-structure', default='DIRECT', type=str)  # {None/'NONE', 'DIRECT'}

    parser.add_argument('--num-particles', default=10, type=int)
    parser.add_argument('--datasets-per-batch', default=16, type=int)
    parser.add_argument('--opt-steps', default=100000, type=int)

    parser.add_argument('--p-lr', default=0.01, type=float)
    parser.add_argument('--q-lr', default=0.01, type=float)
    parser.add_argument('--r-lr', default=0.001, type=float)

    parser.add_argument('--dset-to-plot', default=2, type=int)
    parser.add_argument('--num-val-datasets', default=100, type=int)
    parser.add_argument('--validation-particles', default=250, type=int)
    parser.add_argument('--sweep-test-particles', default=10, type=int)

    parser.add_argument('--load-path', default=None, type=str)  # './params_svm_tmp.p'
    parser.add_argument('--save-path', default=None, type=str)  # './params_svm_tmp.p'

    parser.add_argument('--PLOT', default=1, type=int)

    config = parser.parse_args().__dict__

    # NOTE - we should be able to fork this and use the sample implementation for LDS and SVM.
    # Make sure this one is formatted correctly.
    if 'LDS' in config['model']:
        config['model'] = 'LDS'
    elif 'SVM' in config['model']:
        config['model'] = 'SVM'
    else:
        raise NotImplementedError()

    return config


def svm_define_test(key, free_parameters, proposal_structure, tilt_structure):
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
    true_model, true_states, dataset = svm_define_true_model_and_data(subkey)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = svm_define_test_model(subkey, true_model, free_parameters)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = svm_define_proposal(subkey, model, dataset, proposal_structure)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = svm_define_tilt(subkey, model, dataset, tilt_structure)

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, dataset)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    return ret_model, ret_test, ret_prop, ret_tilt


class SvmTilt(tilts.IndependentGaussianTilt):
    """

    """

    window_length = 2

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

        # Define this for when using the windowed tilt.
        score_criteria = 'window'  # {'all', 'window'}

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = jax.tree_map(lambda args: args[0], params)
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

            # Define the difference scoring criteria.
            def _score_all_future():
                return _idx > t  # Pretty sure this should be a <= (since we are scoring _future_ observations).

            def _score_window():
                return np.logical_and(_score_all_future(), (_idx <= t + SvmTilt.window_length))

            # Decide whether we will score.
            if score_criteria == 'window':
                _score = _score_window()
            elif score_criteria == 'all':
                _score = _score_all_future()
            else:
                raise NotImplementedError()

            return jax.lax.cond(_score,
                                lambda *args: _dist.log_prob(np.asarray([_out])),
                                lambda *args: np.zeros(means.shape[1]),
                                None)

        log_r_val = jax.vmap(_eval)(np.arange(means.shape[0]), means, sds, tilt_outputs).sum(axis=0)

        return log_r_val  # TODO - To disable tilt:  `* 0.0`

    # We need to define the method for generating the inputs.
    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, *_inputs):
        """
        Define the output generator for the svm example.
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


class svmSingleWindowTilt(tilts.IndependentGaussianTilt):

    window_length = 3

    # We need to define the method for generating the inputs.
    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, *_inputs):
        """
        Define the output generator for the svm example.
        Args:
            dataset:

            model:

            particles:

            t:

            *inputs_:

        Returns:

        """

        masked_idx = np.arange(svmSingleWindowTilt.window_length)
        to_insert = (t + 1 + masked_idx < len(dataset))  # We will insert where the window is inside the dataset.

        # Zero out the elements outside of the valid range.
        clipped_dataset = jax.lax.dynamic_slice(dataset,
                                                (t+1, *tuple(0 * _d for _d in dataset.shape[1:])),
                                                (svmSingleWindowTilt.window_length, *dataset.shape[1:]))
        masked_dataset = clipped_dataset * np.expand_dims(to_insert.astype(np.int32), 1)

        # We will pass in whole data into the tilt and then filter out as required.
        tilt_outputs = (masked_dataset, )
        return nn_util.vectorize_pytree(tilt_outputs)


def svm_define_tilt(subkey, model, dataset, tilt_structure):
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

    # # Check whether we have a valid number of tilts.

    # Standard tilt.
    tilt_fn = SvmTilt
    n_tilts = len(dataset[0]) - 1
    tilt_inputs = ()

    # # Single window tilt.
    # tilt_fn = svmSingleWindowTilt
    # n_tilts = 1
    # tilt_inputs = ()

    # Tilt functions take in (dataset, model, particles, t-1).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )

    # Generate the inputs.
    stock_tilt_input = (dataset[-1], model, dummy_particles[0], 0)

    # Generate the outputs.
    dummy_tilt_output = tilt_fn._tilt_output_generator(dataset[-1], model, dummy_particles, 0, *tilt_inputs)

    # Define any custom link functions.
    trunk_fn = None
    head_mean_fn = nn.Dense(dummy_tilt_output.shape[0])
    head_log_var_fn = nn_util.Static(dummy_tilt_output.shape[0])

    # trunk_fn = nn_util.MLP([6, ], output_layer_relu=True)
    # head_mean_fn = nn.Dense(dummy_tilt_output.shape[0])
    # head_log_var_fn = nn.Dense(dummy_tilt_output.shape[0], kernel_init=lambda *args: nn.initializers.lecun_normal()(*args) * 0.01, )

    # Define the tilts themselves.
    tilt = tilt_fn(n_tilts=n_tilts,
                   tilt_input=stock_tilt_input,
                   trunk_fn=trunk_fn,
                   head_mean_fn=head_mean_fn,
                   head_log_var_fn=head_log_var_fn)

    # Initialize the network.
    tilt_params = tilt.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_tilt_fn = tilts.rebuild_tilt(tilt, tilt_structure)
    return tilt, tilt_params, rebuild_tilt_fn


def svm_define_proposal(subkey, model, dataset, proposal_structure):
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

    # trunk_fn = nn_util.MLP([6, ], output_layer_relu=True)
    # head_mean_fn = nn.Dense(dummy_proposal_output.shape[0])
    # head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=lambda *args: nn.initializers.lecun_normal()(*args) * 0.01, )

    # Check whether we have a valid number of proposals.
    n_props = len(dataset[0])

    # Define the required method for building the inputs.
    def svm_proposal_input_generator(_dataset, _model, _particles, _t, _p_dist, _q_state):
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
                                                     input_generator=svm_proposal_input_generator,
                                                     trunk_fn=trunk_fn,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )

    # Initialize the network.
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def svm_get_true_target_marginal(model, data):
    """
    Take in a model and some data and return the tfd distribution representing the marginals of true posterior.
    Args:
        model:
        data:

    Returns:

    """

    try:
        pred_em_posterior = jax.vmap(model.e_step)(data)

        marginal_mean = pred_em_posterior.mean().squeeze()
        marginal_std = np.sqrt(pred_em_posterior.covariance().squeeze())

        pred_em_marginal = tfd.MultivariateNormalDiag(marginal_mean, marginal_std)
    except:
        pred_em_marginal = None

    return pred_em_marginal


def svm_define_true_model_and_data(key):
    """

    Args:
        key:

    Returns:

    """
    num_trials = 100000
    T = 9  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.
    
    # Create the true model.
    key, subkey = jr.split(key)
    true_model = UnivariateSVM()

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=T+1, num_samples=num_trials)

    return true_model, true_states, dataset


def svm_define_test_model(key, true_model, free_parameters):
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
        tmp_model = true_model.__class__()

        # Dig out the free parameters.
        init_free_params = get_free_model_params_fn(tmp_model)

        # Overwrite all the params with the new values.
        default_params = utils.mutate_named_tuple_by_key(true_params, init_free_params)

        # Mutate the free parameters.
        for _k in free_parameters:
            _base = getattr(default_params, _k)
            key, subkey = jr.split(key)
            new_val = {_k: _base + (1.0 * jr.normal(key=subkey, shape=_base.shape))}
            default_params = utils.mutate_named_tuple_by_key(default_params, new_val)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def svm_do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params,
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

                plt.savefig('./svm_param_{}.pdf'.format(_p))

    return param_figs


def svm_do_print(_step, true_model, opt, true_lml, pred_lml, pred_fivo_bound, em_log_marginal_likelihood=None):
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
            print('\tModel:')
            true_str = [_k + ': ' + ' '.join(['{: >5.3f}'.format(_f) for _f in getattr(true_model._parameters, _k).flatten()]) for _k in opt[0].target._fields]
            pred_str = [_k + ': ' + ' '.join(['{: >5.3f}'.format(_f) for _f in getattr(opt[0].target, _k).flatten()]) for _k in opt[0].target._fields]

            print('\t\tTrue: ' + str(true_str))
            print('\t\tPred: ' + str(pred_str))

    # NOTE - the proposal and tilt are more complex here, so don't show them.

    print()
    print()
    print()
