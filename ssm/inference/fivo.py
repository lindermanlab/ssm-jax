"""
FIVO implementation for join state-space inference and parameter learning in SSMs.

FIVO itself isn't that complicated, as it is essentially a wrapper and intuition surrounding a well-implemented SMC
subroutine.  Therefore, what is contained within this script is the basic set of tools required to "run" FIVO as it
is implemented here.  There are a collection of other functions that have to be implemented by the user to make this
code run.  Templates for these other functions are implemented in the accompanying notebook fivo-lds.ipynb .
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from jax import random as jr
from flax import optim
from copy import deepcopy as dc
from ssm.inference.smc import _plot_single_sweep
from tensorflow_probability.substrates.jax import distributions as tfd
import jax.scipy as jscipy

# Import some ssm stuff.
import ssm.utils as utils
from ssm.utils import Verbosity
from ssm.inference.smc import smc

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# Define the standard plotting colours.
color_names = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan"
] * 10


def do_fivo_sweep(_param_vals,
                  _key,
                  _rebuild_model,
                  _rebuild_proposal,
                  _rebuild_tilt,
                  _datasets,
                  _num_particles,
                  **_smc_kw_args):
    """
    Do a single FIVO sweep.  This essentially just wraps a call to the SMC sweep, but where the model and proposal
    are rebuilt on-the-fly from the parameter values passed in.  The returned log-expected-marginal is then imbued with
    the gradient information specified by FIVO.

    Args:
        - _param_vals:              Tuple (or pair) specifying the parameter values for the model and the proposal.
                                    The parameters of the proposal may be `None` if there are no parameters in the
                                    proposal (i.e. if one is using a bootstrap proposal).

        - _key:                     jax.PRNGKey for seeding the sweep.

        - _rebuild_model:           Callable that accepts model parameters (_param_vals[0]) and returns a SSM object.

        - _rebuild_proposal:        Callable that accepts proposal parameters (_param_vals[1]) and returns a callable
                                    that accepts the standard SMC proposal inputs `(dataset, model, particles, time,
                                    p_dist, q_state, ...)`, and returns a distribution over the latent state (z) and
                                    an updated `q_state` (if there is some recurrent state in the proposal).

        - _rebuild_tilt:            Callable that accepts... TODO.

        - _datasets:                Dataset(s) to condition on.

        - _num_particles:           Integer number of particles to use in the sweep.

        - _smc_kw_args:             Keyword arguments to pass into the SMC routine.

    Returns:
        - Tuple: (FIVO-compatible negative log likelihood evaluation, SMCPosterior objects representing sweeps).
    """

    tmp_model = _rebuild_model(_param_vals[0])

    # NOTE - this is a bit sloppy, need to work out if the data is batched in a more reliable way.
    if _datasets.shape[1:] == tmp_model.emissions_shape:
        _smc_posteriors = _do_single_fivo_sweep(_param_vals,
                                                _key,
                                                _rebuild_model,
                                                _rebuild_proposal,
                                                _rebuild_tilt,
                                                _datasets,
                                                _num_particles,
                                                **_smc_kw_args)
    else:
        _single_fivo_sweep_closed = lambda _single_dataset: _do_single_fivo_sweep(_param_vals,
                                                                                  _key,
                                                                                  _rebuild_model,
                                                                                  _rebuild_proposal,
                                                                                  _rebuild_tilt,
                                                                                  _single_dataset,
                                                                                  _num_particles,
                                                                                  **_smc_kw_args)

        _smc_posteriors = jax.vmap(_single_fivo_sweep_closed)(_datasets)

    # Compute the mean of the log marginal.
    _lml = np.mean(_smc_posteriors.log_normalizer)

    return - _lml, _smc_posteriors


def _do_single_fivo_sweep(_param_vals,
                          _key,
                          _rebuild_model,
                          _rebuild_proposal,
                          _rebuild_tilt,
                          _single_dataset,
                          _num_particles,
                          **_smc_kw_args):
    """

    Args:
        _param_vals:
        _key:
        _rebuild_model:
        _rebuild_proposal:
        _rebuild_tilt:
        _single_dataset:
        _num_particles:
        **_smc_kw_args:

    Returns:

    """

    # Reconstruct the model, inscribing the new parameter values.
    _model = _rebuild_model(_param_vals[0])

    # Reconstruct the proposal.
    _proposal = _rebuild_proposal(_param_vals[1], _single_dataset, _model)

    # Build the initial distribution from the zeroth proposal.
    if _proposal is not None:
        initial_distribution = lambda *_args: _proposal(np.zeros(_model.latent_dim, ),
                                                        0,
                                                        _model.initial_distribution(),
                                                        None)
    else:
        initial_distribution = _proposal

    # Reconstruct the tilt.
    _tilt = _rebuild_tilt(_param_vals[2], _single_dataset, _model)

    # Do the sweep.
    _smc_posteriors = smc(_key, _model, _single_dataset,
                          proposal=_proposal,
                          initialization_distribution=initial_distribution,
                          tilt=_tilt,
                          num_particles=_num_particles,
                          **_smc_kw_args)

    return _smc_posteriors


def get_params_from_opt(_opt):
    """
    Pull the parameters (stored in the Flax optimizer target) out of the optimizer tuple.

    If there is an empty optimizer (indicated by None), then None is returned (as this indicates that there are no
    learnable parameters).

    Args:
        - _opt:             Tuple of Flax optimizer objects.

    Returns:
        - Tuple:            Tuple of parameter values defined by the objects in obs (model, proposal).

    """
    return list((_o.target if _o is not None else None) for _o in _opt)


def rebuild_model_fn(_params_in, _default_model):
    """
    Rebuild a new model where default values are specified by `_default_model`, overriden with the values stored in
    _params_in.

    NOTE - there is no seed passed into this function.

    Args:
        - _params_in (NamedTuple):  Named tuple containing the key-value pairs to modify.

    Returns:
        - _model (SSM):             SSM object with

    """

    if _params_in is None:
        return _default_model
    elif len(_params_in) == 0:
        return _default_model

    # NOTE - I think we may want to disable passing in a seed, but I don't think we need
    # to do so.
    # # We cannot pass a new seed into this function or we may get different internal mechanics.
    # assert 'seed' not in _params_in._fields, "[Error]: Cannot pass in a new seed."

    # Get the tuple of parameters used to set up the previous model.
    _default_params = _default_model._parameters

    # Override those parameters with hte new parameters.
    _new_params = utils.mutate_named_tuple_by_key(_default_params, _params_in)

    # Define the new model using the updated params.
    _model = _default_model.__class__(*_new_params)
    return _model


def get_model_params_fn(_model, _keys=None):
    """
    Dig the parameters out of the model.  This is done by accessing the `._parameters` object of cached calling
    arguments.  Specifying `_keys` specified which of these parameters to return as a NamedTuple.

    Args:
        - _model (SSM):                     SSM object from which to extract parameters.
        - _keys (iter[str], optional):      Ordered iterable of strings specifying which keys to retrieve.

    Returns:
        - (NamedTuple):                     Named tuple of the parameters specified by `_keys` or the full set of
                                            calling arguments when `_model` was defined.
    """
    if _keys is not None:
        if len(_keys) == 0:
            return None
    return utils.make_named_tuple(_model._parameters,
                                  keys=_keys,
                                  name=_model._parameters.__class__.__name__ + 'Tmp')


def apply_gradient(full_loss_grad, optimizer):
    """
    Apply the optimization update to the parameters using the gradient.
    `full_loss_grad` and `optimizer` must be tuples of the same pytrees.  I.e., grad[2] will be passed into opt[2].
    The optimizer can be None, in which case there is no gradient update applied.

    Args:
        - full_loss_grad:      Tuple of gradients, each formatted as an arbitrary pytree.
        - optimizer:           Tuple of optimizers, one for each entry in full_loss_grad

    Returns:
        - Updated tuple of optimizers.
    """
    new_optimizer = [(_o.apply_gradient(_g) if _o is not None else None) for _o, _g in zip(optimizer, full_loss_grad)]
    return new_optimizer


def define_optimizer(p_params=None, q_params=None, r_params=None, lr_p=0.001, lr_q=0.001, lr_r=0.001):
    """
    Build out the appropriate optimizer.

    If an inputs is None, then no optimizer is defined and a None flag is used instead.

    Args:
        - p_params (NamedTuple):    Named tuple of the parameters of the SSM.
        - q_params (NamedTuple):    Named tuple of the parameters of the proposal.
        - q_params (NamedTuple):    Named tuple of the parameters of the tilt.
        - p_lr (float):             Float learning rate for p.
        - q_lr (float):             Float learning rate for q.
        - r_lr (float):             Float learning rate for r.

    Returns:
        - (Tuple[opt]):             Tuple of updated optimizers.
    """

    if p_params is not None:
        p_opt_def = optim.Adam(learning_rate=lr_p)
        p_opt = p_opt_def.create(p_params)
    else:
        p_opt = None

    if q_params is not None:
        q_opt_def = optim.Adam(learning_rate=lr_q)
        q_opt = q_opt_def.create(q_params)
    else:
        q_opt = None

    if r_params is not None:
        r_opt_def = optim.Adam(learning_rate=lr_r)
        r_opt = r_opt_def.create(r_params)
    else:
        r_opt = None

    opt = [p_opt, q_opt, r_opt]
    return opt


def compute_single(_tilt_vmapped, _state_single, _obs_single):

    # Compute the tilt value (in log space), remembering that the final state doesn't have a tilt.
    _r_log_val = _tilt_vmapped(_state_single[:-1], np.arange(len(_state_single) - 1), _obs_single)

    return _r_log_val


def compute_elbo(_rebuild_tilt, _tilt_params, _model, _state_batch, _obs_batch):
    """

    Args:
        _rebuild_tilt:
        _vi_opt:
        _model:
        _state_batch:
        _obs_batch:

    Returns:

    """

    # Reconstruct the tilt, but don't bind an observation to it yet.
    _tilt = _rebuild_tilt(_tilt_params, None, _model)

    # Build a tilt function that we can apply at each timestep.
    _tilt_vmapped = jax.vmap(_tilt, in_axes=(0, 0, None))

    # Build a tilt function that we can apply at each timestep.
    _compute_single_vmapped = jax.vmap(compute_single, in_axes=(None, 0, 0))

    # Compute the tilt value (in log space).
    _r_log_vals = _compute_single_vmapped(_tilt_vmapped, _state_batch, _obs_batch)

    # Compute the ELBO as the mean of the tilts.
    _negative_elbo = - np.mean(_r_log_vals)

    return _negative_elbo


def do_vi_tilt_update(key,
                      _env,
                      _param_vals,
                      _rebuild_model,
                      _rebuild_tilt,
                      _state_buffer_raw,
                      _obs_buffer_raw,
                      _vi_opt,
                      _epochs=5,
                      _sgd_batch_size=16):
    print('[test_message]: Hello, im an uncompiled VI update.')
    assert _vi_opt is not None

    # Reconstruct the model, inscribing the current parameter values.
    model = _rebuild_model(_param_vals[0])

    # Construct the batch.
    state_buffer_shaped = np.concatenate(_state_buffer_raw)
    obs_buffer_shaped = np.repeat(np.expand_dims(np.concatenate(_obs_buffer_raw), 1), state_buffer_shaped.shape[1], axis=1)

    state_buffer = state_buffer_shaped.reshape((-1, *state_buffer_shaped.shape[2:]))
    obs_buffer = obs_buffer_shaped.reshape((-1, *obs_buffer_shaped.shape[2:]))

    # Build up the objective function.
    elbo_closed = lambda _p, _x, _y: compute_elbo(_rebuild_tilt, _p, model, _x, _y)
    elbo_val_and_grad = jax.value_and_grad(elbo_closed, argnums=0)

    vi_gradient_steps = 0
    expected_elbo = 0.0

    def _single_epoch(carry, _t):

        (__vi_opt, ) = carry

        state_batch = state_buffer[idxes_batch[_t]]
        obs_batch = obs_buffer[idxes_batch[_t]]

        elbo, grad = elbo_val_and_grad(__vi_opt.target, state_batch, obs_batch)

        __vi_opt = __vi_opt.apply_gradient(grad)

        return (__vi_opt, ), elbo

    # Loop over the epochs.
    for _epoch in range(_epochs):

        # Construct the batches.
        key, subkey = jr.split(key)
        idxes = jr.permutation(subkey, np.arange(len(obs_buffer)))

        if len(idxes) % _sgd_batch_size == 0:
            idxes_trimmed = idxes
        else:
            idxes_trimmed = idxes[0:-(len(idxes) % _sgd_batch_size)]
        idxes_batch = np.reshape(idxes_trimmed, (-1, _sgd_batch_size))

        (_vi_opt, ), elbos = jax.lax.scan(_single_epoch, (_vi_opt, ), (np.arange(len(idxes_batch))))

        vi_gradient_steps += len(idxes_batch)
        expected_elbo = np.mean(elbos)

    return _vi_opt, expected_elbo, vi_gradient_steps


def log_params(_param_hist, _cur_params):
    """
    Parse the parameters and store them for printing.

    Args:
        _param_hist:
        _cur_params:

    Returns:

    """

    # MODEL.
    if _cur_params[0] is not None:
        try:
            _p = _cur_params[0]._asdict()
            _p_flat = {}
            for _k in _p.keys():
                _p_flat[_k] = dc(onp.array(_p[_k].flatten()))
            _param_hist[0].append(_p_flat)
        except:
            print('Logging model parameters failed. ')
            _param_hist[0].append(None)
    else:
        _param_hist[0].append(None)

    # PROPOSAL.
    if _cur_params[1] is not None:
        _p = _cur_params[1]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki

                # _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

                # TODO ---- this is kind of messy.  makes plotting GDM easier but isn't general....
                if ('var' in _k) and ('bias' in _k):
                    _p_flat[_k + '_(EXP)'] = dc(onp.array(np.exp(_p[_ko][_ki])))
                else:
                    _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

        _param_hist[1].append(_p_flat)
    else:
        _param_hist[1].append(None)

    # TILT.
    if _cur_params[2] is not None:
        _p = _cur_params[2]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki

                # _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

                # TODO ---- this is kind of messy.  makes plotting GDM easier but isn't general....
                if ('var' in _k) and ('bias' in _k):
                    _p_flat[_k + '_(EXP)'] = dc(onp.array(np.exp(_p[_ko][_ki])))
                else:
                    _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

        _param_hist[2].append(_p_flat)
    else:
        _param_hist[2].append(None)

    return _param_hist


def initial_validation(env, key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
                       num_particles=1000, dset_to_plot=0, init_model=None, GLOBAL_PLOT=True, do_print=None, do_plot=True):
    """

    Args:
        key:
        true_model:
        dataset:
        true_states:
        opt:
        do_fivo_sweep_jitted:
        _smc_jit:
        num_particles:
        dset_to_plot:
        init_model:

    Returns:

    """
    true_lml, em_log_marginal_likelihood = 0.0, 0.0
    init_bpf_posterior = None
    em_posterior = None
    true_bpf_posterior = None
    true_lml = 0.0
    initial_fivo_bound = 0.0
    init_smc_posterior = None
    initial_lml = 0.0
    em_posterior = None
    em_log_marginal_likelihood = np.nan

    # Test against EM (which for the LDS is exact).
    if hasattr(true_model, 'e_step'):
        em_posterior = jax.vmap(true_model.e_step)(dataset)
        em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
        em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)

    # Test BPF in the true model..
    key, subkey = jr.split(key)
    true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles, resampling_criterion=env.config.resampling_criterion)
    true_lml = - utils.lexp(true_bpf_posterior.log_normalizer)

    if init_model is not None:
        # Test BPF in the initial model..
        key, subkey = jr.split(key)
        init_bpf_posterior = _smc_jit(subkey, init_model, dataset, num_particles=num_particles)
        initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)

    # Test SMC in the initial model.
    key, subkey = jr.split(key)
    initial_fivo_bound, init_smc_posterior = do_fivo_sweep_jitted(subkey, get_params_from_opt(opt),
                                                                  _num_particles=num_particles,
                                                                  _datasets=dataset)
    initial_lml = -utils.lexp(init_smc_posterior.log_normalizer)

    # # Dump any odd and ends of test code in here.
    # temp_validation_code(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
    #                      num_particles=10, dset_to_plot=dset_to_plot, init_model=init_model)

    # TODO - remove this block.  just forcing some EM plotting.
    if em_posterior is not None:
        sweep_em_mean = em_posterior.mean()[dset_to_plot]
        sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset_to_plot]
        sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
        _plot_single_sweep(sweep_em_statistics, true_states[dset_to_plot],
                           tag='EM smoothing', preprocessed=True, obs=dataset[dset_to_plot])

    # Do some plotting.
    if do_plot:
        # if em_posterior is not None:
        #     sweep_em_mean = em_posterior.mean()[dset_to_plot]
        #     sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset_to_plot]
        #     sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
        #     _plot_single_sweep(sweep_em_statistics, true_states[dset_to_plot],
        #                        tag='EM smoothing', preprocessed=True, obs=dataset[dset_to_plot])

        if true_bpf_posterior is not None:
            _plot_single_sweep(true_bpf_posterior[dset_to_plot].filtering_particles,
                               true_states[dset_to_plot],
                               tag='True BPF Filtering (' + str(num_particles) + ' particles).',
                               obs=dataset[dset_to_plot])
            _plot_single_sweep(true_bpf_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
                               true_states[dset_to_plot],
                               tag='True BPF Smoothing (' + str(num_particles) + ' particles).',
                               obs=dataset[dset_to_plot])

        if init_bpf_posterior is not None:
            _plot_single_sweep(init_bpf_posterior[dset_to_plot].filtering_particles,
                               true_states[dset_to_plot],
                               tag='Initial BPF Filtering (' + str(num_particles) + ' particles).',
                               obs=dataset[dset_to_plot])
            _plot_single_sweep(init_bpf_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
                               true_states[dset_to_plot],
                               tag='Initial BPF Smoothing (' + str(num_particles) + ' particles).',
                               obs=dataset[dset_to_plot])

        if init_smc_posterior is not None:
            filt_fig = _plot_single_sweep(init_smc_posterior[dset_to_plot].filtering_particles,
                                          true_states[dset_to_plot],
                                          tag='Initial SMC Filtering (' + str(num_particles) + ' particles).',
                                          obs=dataset[dset_to_plot])
            sweep_fig = _plot_single_sweep(init_smc_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
                                           true_states[dset_to_plot],
                                           tag='Initial SMC Smoothing (' + str(num_particles) + ' particles).',
                                           obs=dataset[dset_to_plot])
        else:
            sweep_fig = None
            filt_fig = None
    else:
        sweep_fig = None
        filt_fig = None

    # Do some print.
    if do_print is not None:
        do_print(0, true_model, opt, true_lml, initial_lml, initial_fivo_bound, em_log_marginal_likelihood)

    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound


def compare_kls(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True, true_bpf_kls=None):
    """

    Args:
        get_marginals:
        env:
        opt:
        dataset:
        true_model:
        rebuild_model_fn:
        rebuild_prop_fn:
        rebuild_tilt_fn:
        key:
        do_fivo_sweep_jitted:
        plot:

    Returns:

    """

    def compute_marginal_kls(smoothing_particles):
        """

        E_q [ log Q / P ]

        Args:
            smoothing_particles:

        Returns:

        """

        # To compute the marginals we are just going to fit a Gaussian.
        kl_p_q = []
        for _t in range(smoothing_particles.shape[-2]):
            samples = smoothing_particles.squeeze()[:, :, _t]
            q_mu = np.mean(samples, axis=1)
            q_sd = np.std(samples, axis=1)  #  + eps

            p_mu = marginals.mean()[:, _t]
            p_sd = marginals.stddev()[:, _t]  # + eps

            _kl_p_q = np.log(q_sd / p_sd) + \
                      (((p_sd ** 2) + ((p_mu - q_mu) ** 2)) / (2.0 * (q_sd ** 2))) + \
                      - 0.5

            _kl_p_q = _kl_p_q.at[q_sd < eps].set(np.nan)

            kl_p_q.append(_kl_p_q)

        return np.asarray(kl_p_q)

    # Set some defaults.
    num_particles = env.config.sweep_test_particles
    eps = 1e-6

    # Get the analytic smoothing marginals.
    marginals = get_marginals(true_model, dataset)

    if marginals is None:
        # TODO - make this more reliable somehow.
        # If there was no analytic marginal available.
        return np.asarray([np.inf]), np.asarray([np.inf])

    # Compare the KLs of the smoothing distributions.
    if true_bpf_kls is None:
        key, subkey = jr.split(key)
        true_bpf_posterior = smc_jitted(subkey, true_model, dataset, num_particles=num_particles)
        true_bpf_kls = compute_marginal_kls(true_bpf_posterior.weighted_smoothing_particles)

    key, subkey = jr.split(key)
    _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
                                                 get_params_from_opt(opt),
                                                 _num_particles=num_particles,
                                                 _datasets=dataset)
    pred_smc_kls = compute_marginal_kls(pred_smc_posterior.weighted_smoothing_particles)

    if plot and env.config.PLOT:
        fig = plt.figure()

        true_median = np.nanquantile(np.asarray(true_bpf_kls), 0.5, axis=1)
        pred_median = np.nanquantile(np.asarray(pred_smc_kls), 0.5, axis=1)

        true_lq = np.nanquantile(np.asarray(true_bpf_kls), 0.25, axis=1)
        pred_lq = np.nanquantile(np.asarray(pred_smc_kls), 0.25, axis=1)

        true_uq = np.nanquantile(np.asarray(true_bpf_kls), 0.75, axis=1)
        pred_uq = np.nanquantile(np.asarray(pred_smc_kls), 0.75, axis=1)

        plt.plot(true_median, label='True (BPF)', c='r')
        plt.plot(pred_median, label='Pred (FIVO-AUX)', c='b')

        plt.plot(true_lq, c='r', linewidth=0.25)
        plt.plot(pred_lq, c='b', linewidth=0.25)

        plt.plot(true_uq, c='r', linewidth=0.25)
        plt.plot(pred_uq, c='b', linewidth=0.25)

        plt.legend()
        plt.grid(True)
        plt.title('E_sweeps [ KL [ p_true[t] || q_pred[t] ] ] (max ' + str(num_particles) + ' particles).  \n' +
                  'NaN KLs (out of ' + str(dataset.shape[0] * dataset.shape[1]) + ') : ' +
                  ' BPF: ' + str(np.sum(np.isnan(true_bpf_kls))) +
                  ' FIVO: ' + str(np.sum(np.isnan(pred_smc_kls))))
        plt.xlabel('Time, t')
        plt.ylabel('KL_t')
        plt.yscale('log')
        plt.pause(0.001)
        plt.savefig('./figs/kl_diff.pdf')
        plt.close(fig)

    return true_bpf_kls, pred_smc_kls


def compare_unqiue_particle_counts(env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True, true_bpf_upc=None):
    """

    Args:
        env:
        opt:
        dataset:
        true_model:
        key:
        do_fivo_sweep_jitted:
        plot:

    Returns:

    """

    def calculate_unique_particle_counts(_particles):
        """
        This is a pain to JAX, so just do it in a loop.

        Args:
            _particles:

        Returns:

        """
        unique_particle_counts = []
        for _sweep in _particles:
            _unique_particle_counts_at_t = []
            for _t in range(_sweep.shape[1]):
                _unique_particle_counts_at_t.append(len(np.unique(_sweep[..., _t, :], axis=0, return_counts=True)[1]))
            unique_particle_counts.append(_unique_particle_counts_at_t)
        return np.asarray(unique_particle_counts)

    num_particles = env.config.sweep_test_particles

    # Compare the KLs of the smoothing distributions.
    if true_bpf_upc is None:
        key, subkey = jr.split(key)
        true_bpf_posterior = smc_jitted(subkey, true_model, dataset, num_particles=num_particles)
        true_bpf_upc = calculate_unique_particle_counts(true_bpf_posterior.weighted_smoothing_particles)

    key, subkey = jr.split(key)
    _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
                                                 get_params_from_opt(opt),
                                                 _num_particles=num_particles,
                                                 _datasets=dataset)
    pred_smc_upc = calculate_unique_particle_counts(pred_smc_posterior.weighted_smoothing_particles)

    if plot and env.config.PLOT:
        fig = plt.figure()
        plt.plot(np.mean(np.asarray(true_bpf_upc), axis=0), label='True (BPF)')
        plt.plot(np.mean(np.asarray(pred_smc_upc), axis=0), label='Pred (FIVO)')

        plt.legend()
        plt.grid(True)
        plt.title(r'E_{sweeps} [ #unique_particles @ t ] (max ' + str(num_particles) + ' particles).')
        plt.xlabel('Time, t')
        plt.ylabel(r'#unique_particles')
        plt.ylim([0.9, num_particles + 0.1])
        plt.plot([0, len(np.mean(np.asarray(true_bpf_upc), axis=0))-1], [1.0, 1.0], c='k', linestyle=':')
        plt.plot([0, len(np.mean(np.asarray(true_bpf_upc), axis=0))-1], [num_particles, num_particles], c='k', linestyle='-.')
        plt.pause(0.001)
        plt.savefig('./figs/ss_diff.pdf')
        plt.close(fig)

    return true_bpf_upc, pred_smc_upc


def compare_sweeps(env, opt, dataset, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key, do_fivo_sweep_jitted, smc_jitted,
                   tag='', nrep=10, true_states=None, num_particles=None):
    """

    Args:
        env:
        opt:
        dataset:
        true_model:
        rebuild_model_fn:
        rebuild_prop_fn:
        rebuild_tilt_fn:
        key:
        do_fivo_sweep_jitted:

    Returns:

    """
    if num_particles is None:
        num_particles = env.config.sweep_test_particles

    # # Do some final validation.
    # # Rebuild the initial distribution.
    # _prop = rebuild_prop_fn(get_params_from_opt(opt)[1])
    # if _prop is not None:
    #     initial_distribution = lambda _dset, _model:  _prop(_dset, _model, np.zeros(dataset.shape[-1], ), 0, _model.initial_distribution(), None)
    # else:
    #     initial_distribution = None

    # BPF in true model.
    key, subkey = jr.split(key)
    final_val_posterior_bpf_true = smc_jitted(subkey,
                                              true_model,
                                              dataset,
                                              num_particles=num_particles)

    # SMC with tilt.
    key, subkey = jr.split(key)
    _, final_val_posterior_fivo_aux = do_fivo_sweep_jitted(subkey,
                                                           get_params_from_opt(opt),
                                                           _num_particles=num_particles,
                                                           _datasets=dataset,)

    # CODE for plotting lineages.
    for _dset_idx in range(nrep):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)
        plt.suptitle('Tag: ' + str(tag) + ', ' + str(num_particles) + ' particles.')

        for _i, _p in enumerate(final_val_posterior_bpf_true[_dset_idx].weighted_smoothing_particles):
            for __i, __p in enumerate(_p.T):
                ax[0].plot(__p, linewidth=0.1, c=color_names[__i], label='Smoothing particles (BPF)' if ((__i == 0) and (_i == 0)) else None)
        ax[0].grid(True)
        ax[0].set_title('BPF in true model')

        for _i, _p in enumerate(final_val_posterior_fivo_aux[_dset_idx].weighted_smoothing_particles):
            for __i, __p in enumerate(_p.T):
                ax[1].plot(__p, linewidth=0.1, c=color_names[__i], label='Smoothing particles (FIVO)' if ((__i == 0) and (_i == 0)) else None)
        ax[1].grid(True)

        # # Plot the observed data.
        # for _i, _p in enumerate(dataset[_dset_idx].T):
        #     ax[0].plot(_p, linewidth=1.0, c=color_names[_i], linestyle=':', label='Observed data' if _i == 0 else None)
        #     ax[1].plot(_p, linewidth=1.0, c=color_names[_i], linestyle=':', label='Observed data' if _i == 0 else None)

        if (get_params_from_opt(opt)[1] is not None) and (get_params_from_opt(opt)[2] is not None):
            ax[1].set_title('SMC-AUX with learned pqr.')
            _tag = 'pqr'
        elif (get_params_from_opt(opt)[1] is not None) and (get_params_from_opt(opt)[2] is None):
            ax[1].set_title('SMC-AUX with learned pq.')
            _tag = 'pq'
        elif (get_params_from_opt(opt)[1] is None) and (get_params_from_opt(opt)[2] is not None):
            ax[1].set_title('SMC-AUX with learned pr.')
            _tag = 'pr'
        else:
            ax[1].set_title('SMC-AUX with learned p...?')
            _tag = 'p'

        if true_states is not None:

            if len(true_states.shape) == 2:
                _true_states = true_states
            else:
                _true_states = true_states[_dset_idx]

            for _i, _p in enumerate(_true_states.T):
                ax[0].plot(_p, linewidth=1.0, c=color_names[_i], linestyle='--', label='True states' if _i == 0 else None)
                ax[1].plot(_p, linewidth=1.0, c=color_names[_i], linestyle='--', label='True states' if _i == 0 else None)

        ax[1].legend()

        plt.pause(0.01)
        plt.savefig('./figs/tmp_sweep_{}_{}.pdf'.format(_tag, _dset_idx))
        plt.close(fig)


def final_validation(get_marginals,
                     env,
                     opt,
                     dataset,
                     true_model,
                     rebuild_model_fn,
                     rebuild_prop_fn,
                     rebuild_tilt_fn,
                     key,
                     do_fivo_sweep_jitted,
                     smc_jitted,
                     GLOBAL_PLOT=True,
                     tag=''):
    """

    Args:
        get_marginals:
        env:
        opt:
        dataset:
        true_model:
        rebuild_model_fn:
        rebuild_prop_fn:
        rebuild_tilt_fn:
        key:
        do_fivo_sweep_jitted:

    Returns:

    """

    # Compare the sweeps.
    compare_sweeps(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, tag=tag)

    # Compare the KLs.
    true_bpf_kls, pred_smc_kls = compare_kls(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted,
                                             plot=True)


# def temp_validation_code(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
#                          num_particles=10, dset_to_plot=0, init_model=None):
#     """
#
#     Args:
#         key:
#         true_model:
#         dataset:
#         true_states:
#         opt:
#         do_fivo_sweep_jitted:
#         _smc_jit:
#         num_particles:
#         dset_to_plot:
#         init_model:
#
#     Returns:
#
#     """
#
#     # Do some sweeps.
#     key, subkey = jr.split(key)
#     smc_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
#     key, subkey = jr.split(key)
#     initial_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey, get_params_from_opt(opt),
#                                                                 num_particles=num_particles,
#                                                                 datasets=dataset)
#
#     # CODE for plotting lineages.
#     idx = 7
#     fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)
#     for _p in smc_posterior[idx].weighted_smoothing_particles:
#         ax[0].plot(_p, linewidth=0.1, c='b')
#     ax[0].grid(True)
#     for _p in sweep_posteriors[idx].weighted_smoothing_particles:
#         ax[1].plot(_p, linewidth=0.1, c='b')
#     ax[1].grid(True)
#     plt.pause(0.01)
#
#     # Compare the variances of the LML estimates.
#     # Test BPF in the initial model..
#     val_bpf_lml, val_fivo_lml = [], []
#     for _ in range(20):
#         key, subkey = jr.split(key)
#         true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
#         true_bpf_lml = - utils.lexp(true_bpf_posterior.log_normalizer)
#         val_bpf_lml.append(true_bpf_lml)
#
#     for _ in range(20):
#         key, subkey = jr.split(key)
#         initial_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey, get_params_from_opt(opt),
#                                                                     num_particles=num_particles,
#                                                                     datasets=dataset)
#         initial_lml = -utils.lexp(sweep_posteriors.log_normalizer)
#         val_fivo_lml.append(initial_lml)
#
#     print('Variance: BPF:      ', np.var(np.asarray(val_bpf_lml)))
#     print('Variance: FIVO-AUX: ', np.var(np.asarray(val_fivo_lml)))

# def compare_ess(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True, true_bpf_ess=None):
#     """
#
#     Args:
#         get_marginals:
#         env:
#         opt:
#         dataset:
#         true_model:
#         key:
#         do_fivo_sweep_jitted:
#         smc_jitted:
#         plot:
#         true_bpf_ess:
#
#     Returns:
#
#     """
#
#     def compute_ess(smoothing_particles):
#         """
#
#
#
#         Args:
#             get_marginals:
#             true_model:
#             dataset:
#             smoothing_particles:
#
#         Returns:
#
#         """
#
#         # To compute the marginals we are just going to fit a Gaussian.
#         ess = []
#         for _t in range(smoothing_particles.shape[-2]):
#             samples = smoothing_particles.squeeze()[:, :, _t]
#
#             p_mu = marginals.mean()[:, _t]
#             p_sd = marginals.stddev()[:, _t]
#
#             # Evaluate the probability of the particle sets under each marginal.
#             eval = jax.vmap(lambda _mu, _sd, _s: tfd.MultivariateNormalDiag(np.expand_dims(_mu, 0), np.expand_dims(_sd, 0)).prob(np.expand_dims(_s, 1)))
#             weights = eval(p_mu, p_sd, samples)
#
#             _ess = np.square(np.sum(weights, axis=1)) / np.sum(np.square(weights), axis=1)
#
#             ess.append(_ess)
#
#         return np.asarray(ess)
#
#     # Set some defaults.
#     num_particles = env.config.sweep_test_particles
#
#     # Get the analytic smoothing marginals.
#     marginals = get_marginals(true_model, dataset)
#
#     if marginals is None:
#         # TODO - make this more reliable somehow.
#         # If there was no analytic marginal available.
#         return np.asarray([np.inf])
#
#     # Compare the KLs of the smoothing distributions.
#     if true_bpf_ess is None:
#         key, subkey = jr.split(key)
#         true_bpf_posterior = smc_jitted(subkey, true_model, dataset, num_particles=num_particles)
#         true_bpf_ess = compute_ess(true_bpf_posterior.weighted_smoothing_particles)
#
#     key, subkey = jr.split(key)
#     _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
#                                                  get_params_from_opt(opt),
#                                                  _num_particles=num_particles,
#                                                  _datasets=dataset)
#     pred_smc_ess = compute_ess(pred_smc_posterior.weighted_smoothing_particles)
#
#     if plot and env.config.PLOT:
#         fig = plt.figure()
#         plt.plot(np.mean(np.asarray(true_bpf_ess), axis=1), label='True (BPF)')
#         plt.plot(np.mean(np.asarray(pred_smc_ess), axis=1), label='Pred (FIVO-AUX)')
#         # plt.plot(np.median(np.asarray(init_bpf_kls), axis=1), label='bpf')
#         plt.legend()
#         plt.grid(True)
#         plt.title('E_sweeps [ ess_t ] (max ' + str(num_particles) + ' particles).')
#         plt.xlabel('Time, t')
#         plt.ylabel('ESS_t')
#         plt.ylim([0.9, num_particles + 0.1])
#         plt.plot([0, len(np.mean(np.asarray(true_bpf_ess), axis=1))-1], [1.0, 1.0], c='k', linestyle=':')
#         plt.plot([0, len(np.mean(np.asarray(true_bpf_ess), axis=1))-1], [num_particles, num_particles], c='k', linestyle='-.')
#         plt.pause(0.001)
#         plt.savefig('./figs/ESS_diff.pdf')
#         plt.close(fig)
#
#     return true_bpf_ess, pred_smc_ess