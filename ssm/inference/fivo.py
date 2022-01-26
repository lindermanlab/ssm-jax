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
from types import SimpleNamespace
import wandb
from pprint import pprint
import git

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


def do_fivo_config(DEFAULT_MODEL, USE_WANDB, PROJECT, USERNAME, LOCAL_SYSTEM):
    """

    Args:
        DEFAULT_MODEL:
        USE_WANDB:
        PROJECT:
        USERNAME:
        LOCAL_SYSTEM:

    Returns:

    """

    # Quickly hack finding the model and importing the right config.
    import sys
    try:
        model = sys.argv[np.where(np.asarray([_a == '--model' for _a in sys.argv]))[0][0] + 1]
    except:
        model = DEFAULT_MODEL
        print('[WARNING]: No model specified, defaulting to: ', model)

    # Import the config for the specified model.
    if 'LDS' in model:
        from ssm.inference.models.test_fivo_lds import get_config
    elif 'GDM' in model:
        from ssm.inference.models.test_fivo_gdm import get_config
    elif 'SVM' in model:
        from ssm.inference.models.test_fivo_svm import get_config
    elif 'VRNN' in model:
        from ssm.inference.models.test_fivo_vrnn import get_config
    else:
        raise NotImplementedError()

    # Go and get the model-specific config.
    config, do_print, define_test, do_plot, get_marginals = get_config()

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    if config['free_parameters'] is None or config['free_parameters'] == '':
        config['free_parameters'] = ()
    else:
        config['free_parameters'] = tuple(config['free_parameters'].replace(' ', '').split(','))

    # Force the tilt temperature to zero if we are not using tilts.  this is just bookkeeping, really.
    if config['tilt_structure'] == 'NONE' or config['tilt_structure'] is None:
        config['temper'] = 0.0

    # Get everything.
    if USE_WANDB:
        # Set up WandB
        env = wandb.init(project=PROJECT, entity=USERNAME, group=config['log_group'], config=config)
    else:
        log_group = 'none'
        env = SimpleNamespace(**{'config': SimpleNamespace(**config),
                                 'log_group': log_group})

    # Set up some WandB stuff.
    env.config.wandb_group = env.config.log_group
    env.config.use_wandb = bool(USE_WANDB)
    env.config.wandb_project = PROJECT
    env.config.local_system = LOCAL_SYSTEM

    # Grab some git information.
    git_commit = 'NoneFound'
    git_branch = 'NoneFound'
    git_is_dirty = 'NoneFound'
    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit = repo.head.object.hexsha
        git_branch = repo.active_branch
        git_is_dirty = repo.is_dirty()
    except:
        print('[WARNING]: Failed to grab git info...')
        env.config.git_commit = git_commit
        env.config.git_branch = git_branch
        env.config.git_is_dirty = git_is_dirty

    # Set up the first key
    key = jr.PRNGKey(env.config.seed)

    # Do some final bits.
    if len(env.config.free_parameters) == 0: print('\n[WARNING]: NO FREE MODEL PARAMETERS...\n')
    pprint(env.config)
    return env, key, do_print, define_test, do_plot, get_marginals


def do_fivo_sweep(_param_vals,
                  _key,
                  _rebuild_model,
                  _rebuild_proposal,
                  _rebuild_tilt,
                  _datasets,
                  _masks,
                  _num_particles,
                  _use_bootstrap_initial_distribution,
                  **_smc_kw_args):
    """
    Do a single FIVO sweep.  This essentially just wraps a call to the SMC sweep, but where the model and proposal
    are rebuilt on-the-fly from the parameter values passed in.  The returned log-expected-marginal is then imbued with
    the gradient information specified by FIVO.

    Args:
        _param_vals:              Tuple (or pair) specifying the parameter values for the model and the proposal.
                                  The parameters of the proposal may be `None` if there are no parameters in the
                                  proposal (i.e. if one is using a bootstrap proposal).

        _key:                     jax.PRNGKey for seeding the sweep.

        _rebuild_model:           Callable that accepts model parameters (_param_vals[0]) and returns a SSM object.

        _rebuild_proposal:        Callable that accepts proposal parameters (_param_vals[1]) and returns a callable
                                  that accepts the standard SMC proposal inputs `(dataset, model, particles, time,
                                  p_dist, q_state, ...)`, and returns a distribution over the latent state (z) and
                                  an updated `q_state` (if there is some recurrent state in the proposal).

        _rebuild_tilt:            Callable that accepts... TODO.

        _datasets:                Dataset(s) to condition on.

        _num_particles:           Integer number of particles to use in the sweep.

        _use_bootstrap_initial_distribution: Set to true to overwride other behaviours and use the model for initialization.

        _smc_kw_args:             Keyword arguments to pass into the SMC routine.

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
                                                _masks,
                                                _num_particles,
                                                _use_bootstrap_initial_distribution,
                                                **_smc_kw_args)
    else:
        _single_fivo_sweep_closed = lambda _single_dataset, _single_mask: _do_single_fivo_sweep(_param_vals,
                                                                                                _key,
                                                                                                _rebuild_model,
                                                                                                _rebuild_proposal,
                                                                                                _rebuild_tilt,
                                                                                                _single_dataset,
                                                                                                _single_mask,
                                                                                                _num_particles,
                                                                                                _use_bootstrap_initial_distribution,
                                                                                                **_smc_kw_args)

        _smc_posteriors = jax.vmap(_single_fivo_sweep_closed)(_datasets, _masks)

    # Compute the mean of the log marginal.
    _lml = np.mean(_smc_posteriors.log_normalizer)

    return - _lml, _smc_posteriors


def _do_single_fivo_sweep(_param_vals,
                          _key,
                          _rebuild_model,
                          _rebuild_proposal,
                          _rebuild_tilt,
                          _single_dataset,
                          _single_mask,
                          _num_particles,
                          _use_bootstrap_initial_distribution,
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
        _use_bootstrap_initial_distribution:
        **_smc_kw_args:

    Returns:

    """

    # Reconstruct the model, inscribing the new parameter values.
    _model = _rebuild_model(_param_vals[0])

    # Reconstruct the proposal.
    _proposal = _rebuild_proposal(_param_vals[1], _single_dataset, _model)

    # Build the initial distribution from the zeroth proposal.
    if (_proposal is None) or _use_bootstrap_initial_distribution:
        initial_distribution = None
    else:
        initial_distribution = lambda *_args: _proposal(np.zeros(_model.latent_dim, ),
                                                        0,
                                                        _model.initial_distribution(),
                                                        None)

    # Reconstruct the tilt.
    _tilt = _rebuild_tilt(_param_vals[2], _single_dataset, _model)

    # Do the sweep.
    _smc_posteriors = smc(_key,
                          _model,
                          _single_dataset,
                          masks=_single_mask,
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
        _opt:             Tuple of Flax optimizer objects.

    Returns:
        Tuple:            Tuple of parameter values defined by the objects in obs (model, proposal).

    """
    return list((_o.target if _o is not None else None) for _o in _opt)


def rebuild_model_fn(_params_in, _default_model):
    """
    Rebuild a new model where default values are specified by `_default_model`, overriden with the values stored in
    _params_in.

    NOTE - there is no seed passed into this function.

    Args:
        _params_in (NamedTuple):  Named tuple containing the key-value pairs to modify.

        _default_model:

    Returns:
        _model (SSM):             SSM object with

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
        _model (SSM):                     SSM object from which to extract parameters.
        _keys (iter[str], optional):      Ordered iterable of strings specifying which keys to retrieve.

    Returns:
        (NamedTuple):                     Named tuple of the parameters specified by `_keys` or the full set of
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
        full_loss_grad:      Tuple of gradients, each formatted as an arbitrary pytree.
        optimizer:           Tuple of optimizers, one for each entry in full_loss_grad

    Returns:
        Updated tuple of optimizers.
    """
    new_optimizer = [(_o.apply_gradient(_g) if _o is not None else None) for _o, _g in zip(optimizer, full_loss_grad)]
    return new_optimizer


def define_optimizer(p_params=None, q_params=None, r_params=None, lr_p=0.001, lr_q=0.001, lr_r=0.001):
    """
    Build out the appropriate optimizer.

    If an inputs is None, then no optimizer is defined and a None flag is used instead.

    Args:
        p_params (NamedTuple):    Named tuple of the parameters of the SSM.
        q_params (NamedTuple):    Named tuple of the parameters of the proposal.
        q_params (NamedTuple):    Named tuple of the parameters of the tilt.
        lr_p (float):             Float learning rate for p.
        lr_q (float):             Float learning rate for q.
        lr_r (float):             Float learning rate for r.

    Returns:
        (Tuple[opt]):             Tuple of updated optimizers.
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

