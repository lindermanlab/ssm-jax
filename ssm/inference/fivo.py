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

    # Reconstruct the tilt.
    _tilt = _rebuild_tilt(_param_vals[2])

    # Build up the initial distribution using some dummy particles with all zeros.
    if _param_vals[1] is not None:
        _dummy_particles = _model.initial_distribution().sample(seed=_key, batch_shape=(_num_particles, ))
        _dummy_particles = jax.tree_map(lambda arg: 0.0*arg, _dummy_particles)
        _initial_dist = lambda _dataset, _model: _proposal(_dataset,
                                                           _model,
                                                           _dummy_particles,
                                                           0,
                                                           _model.initial_distribution(),
                                                           None)
    else:
        _initial_dist = None

    # Do the sweep.
    _smc_posteriors = smc(_key, _model, _dataset,
                          proposal=_proposal,
                          tilt=_tilt,
                          num_particles=_num_particles,
                          initialization_distribution=_initial_dist,
                          **_smc_kw_args)

    # Compute the log of the expected marginal.
    # TODO - this should take the mean of the log normalizers in FIVO, but this isn't actually the expected log
    #  likelihood...
    # _lml = utils.lexp(_smc_posteriors.log_normalizer)
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
    return tuple((_o.target if _o is not None else None) for _o in _opt)


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

    # We cannot pass a new seed into this function or we may get different internal mechanics.
    assert 'seed' not in _params_in._fields, "[Error]: Cannot pass in a new seed."

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


def define_optimizer(p_params=None, q_params=None, r_params=None, p_lr=0.001, q_lr=0.001, r_lr=0.001):
    """
    Build out the appropriate optimizer.

    If an inputs is None, then no optimizer is defined and a None flag is used instead.

    Args:
        - p_params (NamedTuple):    Named tuple of the parameters of the SSM.
        - q_params (NamedTuple):    Named tuple of the parameters of the proposal.
        - r_params (NamedTuple):    Named tuple of the parameters of the tilt.
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

