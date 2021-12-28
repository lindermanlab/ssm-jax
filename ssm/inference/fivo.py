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
from jax import random as jr
from flax import optim
from copy import deepcopy as dc
import numpy as onp

# Import some ssm stuff.
from ssm.utils import Verbosity
from ssm.inference.smc import smc
import ssm.utils as utils

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def do_fivo_sweep(_param_vals,
                  _key,
                  _rebuild_model,
                  _rebuild_proposal,
                  _rebuild_tilt,
                  _dataset,
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

        - _dataset:                 Dataset(s) to condition on.

        - _num_particles:           Integer number of particles to use in the sweep.

        - _smc_kw_args:             Keyword arguments to pass into the SMC routine.

    Returns:
        - Tuple: (FIVO-compatible negative log likelihood evaluation, SMCPosterior objects representing sweeps).
    """

    # Reconstruct the model, inscribing the new parameter values.
    _model = _rebuild_model(_param_vals[0])

    # Reconstruct the proposal.
    _proposal = _rebuild_proposal(_param_vals[1])

    # Build the initial distribution from the zeroth proposal.
    if _proposal is not None:
        initial_distribution = lambda _dset, _model: _proposal(_dset,
                                                               _model,
                                                               np.zeros(_dataset.shape[-1], ),
                                                               0,
                                                               None,
                                                               None)
    else:
        initial_distribution = _proposal

    # Reconstruct the tilt.
    _tilt = _rebuild_tilt(_param_vals[2])

    # Do the sweep.
    _smc_posteriors = smc(_key, _model, _dataset,
                          proposal=_proposal,
                          initialization_distribution=initial_distribution,
                          tilt=_tilt,
                          num_particles=_num_particles,
                          **_smc_kw_args)

    # Compute the mean of the log marginal.
    _lml = np.mean(_smc_posteriors.log_normalizer)

    return - _lml, _smc_posteriors


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


def define_optimizer(p_params=None, q_params=None, r_params=None, p_lr=0.01, q_lr=0.01, r_lr=0.01):
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
        p_opt_def = optim.Adam(learning_rate=p_lr)
        p_opt = p_opt_def.create(p_params)
    else:
        p_opt = None

    if q_params is not None:
        q_opt_def = optim.Adam(learning_rate=q_lr)
        q_opt = q_opt_def.create(q_params)
    else:
        q_opt = None

    if r_params is not None:
        r_opt_def = optim.Adam(learning_rate=r_lr)
        r_opt = r_opt_def.create(r_params)
    else:
        r_opt = None

    opt = [p_opt, q_opt, r_opt]
    return opt


def log_params(_param_hist, _cur_params, _cur_lml, _cur_fivo, _cur_em, _step):
    """
    Parse the parameters and store them for printing.

    Args:
        _param_hist:
        _cur_params:

    Returns:

    """

    # MODEL.
    if _cur_params[0] is not None:
        _p = _cur_params[0]._asdict()
        _p_flat = {}
        for _k in _p.keys():
            _p_flat[_k] = dc(onp.array(_p[_k].flatten()))
        _param_hist[0].append(_p_flat)
    else:
        _param_hist[0].append(None)

    # _prop_dict = {'head_mean_kernel': _cur_params[1]._dict['params']['head_mean_fn']['kernel'].flatten(),
    #               'head_mean_bias': _cur_params[1]._dict['params']['head_mean_fn']['bias'].flatten(),
    #               'head_var_bias': np.exp(_cur_params[1]._dict['params']['head_log_var_fn']['bias'])}
    # _param_hist[1].append(_prop_dict)
    #
    # _tilt_dict = {'head_mean_kernel': _cur_params[2]._dict['params']['head_mean_fn']['kernel'].flatten(),
    #               'head_mean_bias': _cur_params[2]._dict['params']['head_mean_fn']['bias'].flatten(),
    #               'head_var_bias': np.exp(_cur_params[2]._dict['params']['head_log_var_fn']['bias'])}
    # _param_hist[2].append(_tilt_dict)

    # PROPOSAL.
    if _cur_params[1] is not None:
        _p = _cur_params[1]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki

                # _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

                # TODO - ----
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

                # TODO - ----
                if ('var' in _k) and ('bias' in _k):
                    _p_flat[_k + '_(EXP)'] = dc(onp.array(np.exp(_p[_ko][_ki])))
                else:
                    _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

        _param_hist[2].append(_p_flat)
    else:
        _param_hist[2].append(None)

    # Add the loss terms.
    _param_hist[3].append(dc(_cur_lml))
    _param_hist[4].append(dc(_cur_fivo))
    _param_hist[5].append(dc(_cur_em))
    _param_hist[6].append(dc(_step))

    return _param_hist

