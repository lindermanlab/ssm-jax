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
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import NamedTuple, Any
from flax import optim
from copy import deepcopy as dc

# Import some ssm stuff.
from ssm.utils import Verbosity
from ssm.inference.smc import smc
import ssm.utils as utils

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def do_fivo_sweep(_param_vals, _key, _rebuild_model, _rebuild_proposal, _dataset, _num_particles):
    """

    :param _param_vals:
    :param _key:
    :param _rebuild_model:
    :param _rebuild_proposal:
    :param _dataset:
    :param _num_particles:
    :return:
    """

    # Reconstruct the model, inscribing the new parameter values.
    _model = _rebuild_model(_param_vals[0])

    # Reconstruct the proposal.
    _proposal = _rebuild_proposal(_param_vals[1])

    # Do the sweep.
    _smc_posteriors = smc(_key, _model, _dataset, proposal=_proposal, num_particles=_num_particles)

    # Compute the log of the expected marginal.
    _lml = utils.lexp(_smc_posteriors.log_normalizer)

    return - _lml, _smc_posteriors


# def define_rebuild_model(_model, _p_params_accessors):
#     """
#     This function can take anything as arguments, but MUST return a function that takes EXACTLY the parameters of the
#     model and in turn returns the model updated with the supplied parameters..
#
#     # TODO - this paradigm may need updating.
#
#     :param _model:
#     :param _p_params_accessors:
#     :return:
#     """
#
#     def rebuild_model(_param_vals):
#         _rebuilt_model = dc(_model)
#         for _v, _a in zip(_param_vals, _p_params_accessors):
#             _rebuilt_model = _a(_rebuilt_model, _v)
#         return _rebuilt_model
#
#     return rebuild_model


# def get_params_from_model(model, accessors):
#     """
#
#     :param model:
#     :param accessors:
#     :return:
#     """
#     p_params = tuple(_a(model) for _a in accessors)
#     return p_params


def get_params_from_opt(_opt):
    """
    Pull the parameters (stored in the Flax optimizer target) out of the optimizer tuple.
    :param _opt: Tuple of Flax optimizer objects.
    :return: Tuple of parameters.
    """
    return tuple((_o.target if _o is not None else None) for _o in _opt)


def apply_gradient(full_loss_grad, optimizer):
    """
    Apply the optimization update to the parameters using the gradient.

    full_loss_grad and optimizer must be tuples of the same pytrees.  I.e., grad[4] will be passed into opt[4].

    The optimizer can be None, in which case there is no gradient update applied.

    :param full_loss_grad:      Tuple of gradients, each formatted as an arbitrary pytree.
    :param optimizer:           Tuple of optimizers, one for each entry in full_loss_grad
    :return:                    Updated tuple of optimizers.
    """
    new_optimizer = [(_o.apply_gradient(_g) if _o is not None else None) for _o, _g in zip(optimizer, full_loss_grad)]
    return new_optimizer


def define_optimizer(p_params=None, q_params=None, p_lr=0.001, q_lr=0.001):
    """
    Build out the appropriate optimizer.

    If an inputs is None, then no optimizer is defined and a None flag is used instead.

    :param p_params:    Pytree of the parameters of the SSM.
    :param q_params:    PyTree of the parameters of the proposal.
    :param p_lr:        Float learning rate for p.
    :param q_lr:        Float learning rate for q.
    :return:
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

    opt = [p_opt, q_opt]
    return opt

