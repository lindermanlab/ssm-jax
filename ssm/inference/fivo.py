"""
FIVO implementation for join state-space inference and parameter learning in SSMs.
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
from ssm.utils import Verbosity, random_rotation
from ssm.inference.smc import smc, plot_single_sweep
from ssm.inference.em import em
import ssm.distributions as ssmd
from ssm.inference.conditional_generators import build_independent_gaussian_generator
# import ssm.snax.snax as snax
import flax.linen as nn
from ssm.lds.models import GaussianLDS
import ssm.nn_util as nn_util

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# NOTE - this is really useful, but will break parts of the
# computation graph if NaNs are used for signalling purposes.
# NaN debugging stuff.
from jax.config import config
config.update("jax_debug_nans", True)

disable_jit = False
from contextlib import contextmanager, ExitStack
@contextmanager
def nothing():
    yield
possibly_disabled = jax.disable_jit if disable_jit else nothing


def get_params(_opt):
    """
    Pull the parameters (stored in the Flax optimizer target) out of the optimizer tuple.
    :param _opt: Tuple of Flax optimizer objects.
    :return: Tuple of parameters.
    """
    return tuple((_o.target if _o is not None else None) for _o in _opt)


def lexp(_lmls):
    """
    Compute the log-expectation of a ndarray of log probabilities.
    :param _lmls:
    :return:
    """
    _lml = spsp.logsumexp(_lmls) - len(_lmls)
    return _lml


def do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood=None):
    """
    Do menial print stuff.
    :param _step:
    :param pred_lml:
    :param true_model:
    :param true_lml:
    :param opt:
    :param em_log_marginal_likelihood:
    :return:
    """

    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},  Pred Neg-LML: {: >8.3f}'.format(_step, true_lml, pred_lml)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    print('True: dynamics:  ', '  '.join(['{: >9.5f}'.format(_s) for _s in true_model.dynamics_matrix.flatten()]))
    print('Pred: dynamics:  ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[0].target[0].flatten()]))

    # if opt[1] is not None:
    #     print('True: log-var:   ', '  '.join(['{: >9.5f}'.format(_s) for _s in np.log(np.diagonal(true_model.dynamics_noise_covariance))]))
    #     print('Pred: q log-var: ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[1].target._asdict()['head_log_var_fn'].W.flatten()]))
    print()


def initial_validation(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted):
    """
    Do an test of the true model and the initialized model.
    :param key:
    :param true_model:
    :param dataset:
    :param true_states:
    :param opt:
    :param _do_fivo_sweep_jitted:
    :return:
    """
    true_lml, em_log_marginal_likelihood = 0.0, 0.0

    # Test against EM (which for the LDS is exact).
    em_posterior = jax.vmap(true_model.infer_posterior)(dataset)
    em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    em_log_marginal_likelihood = - lexp(em_log_marginal_likelihood)
    sweep_em_mean = em_posterior.mean()[0]
    sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[0]
    sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    plot_single_sweep(sweep_em_statistics, true_states[0], tag='EM smoothing', preprocessed=True)

    # Test SMC in the true model..
    key, subkey = jr.split(key)
    true_sweep, true_lml, _, _ = smc(subkey, true_model, dataset, num_particles=5000)
    true_lml = - lexp(true_lml)
    plot_single_sweep(true_sweep[0], true_states[0], tag='True Smoothing.')

    # Test SMC in the initial model.
    initial_params = dc(get_params(opt))
    key, subkey = jr.split(key)
    initial_lml, initial_sweep = _do_fivo_sweep_jitted(subkey, get_params(opt), _num_particles=5000)
    sweep_fig = plot_single_sweep(initial_sweep[0], true_states[0], tag='Initial Smoothing.')
    do_print(0, initial_lml, true_model, true_lml, opt, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig


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


def define_optimizer(p_params=None, q_params=None, r_params=None):
    """
    Build out the appropriate optimizer.

    If an inputs is None, then no optimizer is defined and a None flag is used instead.

    :param p_params:    Pytree of the parameters of the SSM.
    :param q_params:    PyTree of the parameters of the proposal.
    :param r_params:    PyTree of the parameters of the tilt.
    :return:
    """

    if p_params is not None:
        p_opt_def = optim.Adam(learning_rate=0.001)
        p_opt = p_opt_def.create(p_params)
    else:
        p_opt = None

    if q_params is not None:
        q_opt_def = optim.Adam(learning_rate=0.0001)
        q_opt = q_opt_def.create(q_params)
    else:
        q_opt = None

    if r_params is not None:
        r_opt_def = optim.Adam(learning_rate=0.0001)
        r_opt = r_opt_def.create(r_params)
    else:
        r_opt = None

    opt = [p_opt, q_opt, r_opt]
    return opt


class independent_gaussian_proposal():
    """
    Define a proposal FUNCTION that is an indpendent gaussian.  This Module is actually just a thin wrapper over a set
    of Linen modules.

    To define a different proposal FUNCTION here (as opposed to a proposal structure), change this class.

    This modules `.apply` method also wraps a call to the input generator, which takes the standard proposal
    parametersization (dataset, model, particles, t, p_dist, q_state) and flattens it into the right form.

    :param n_proposals:
    :param stock_proposal_input_without_q_state:
    :param dummy_output:
    :param trunk_fn:
    :param head_mean_fn:
    :param head_log_var_fn:
    :return:
    """

    def __init__(self, n_proposals, stock_proposal_input_without_q_state, dummy_output,
                 trunk_fn=None, head_mean_fn=None, head_log_var_fn=None):

        assert n_proposals == 1, 'Can only use a single proposal.'

        # Re-build the full input that will be provided.
        q_state = None
        full_input = (*stock_proposal_input_without_q_state, q_state)

        # Parse the input.
        self._dummy_processed_input = self.proposal_input_generator(*full_input)

        # Build out the function approximator.
        self.proposal = build_independent_gaussian_generator(self._dummy_processed_input,
                                                             dummy_output,
                                                             trunk_fn=trunk_fn,
                                                             head_mean_fn=head_mean_fn,
                                                             head_log_var_fn=head_log_var_fn, )

    def init(self, key):
        return self.proposal.init(key, self._dummy_processed_input)

    def apply(self, params, inputs):
        proposal_inputs = self.proposal_input_generator(*inputs)
        q_dist = self.proposal.apply(params, proposal_inputs)
        return q_dist, None

    def proposal_input_generator(self, *_inputs):
        """
        Inputs of the form: (dataset, model, particle[SINGLE], t, p_dist, q_state).
        :param _inputs:
        :return:
        """

        dataset, _, particles, t, _, _ = _inputs  # NOTE - this part of q can't actually use model or p_dist.
        proposal_inputs = (jax.lax.dynamic_index_in_dim(_inputs[0], index=0, axis=0, keepdims=False), _inputs[2])

        is_batched = (_inputs[1].latent_dim != particles.shape[0])
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)


def define_proposal(n_proposals, stock_proposal_input_without_q_state, dummy_output):
    """
    Define the proposal.  Right now, this is implemented to define a static proposal with no input dependence.

    This function would be then changed to define more expressive proposals.

    :param n_proposals:
    :param stock_proposal_input_without_q_state:
    :param dummy_output:
    :return:
    """

    # Define a more conservative initialization.
    w_init_mean = lambda *args: (0.01 * jax.nn.initializers.glorot_normal()(*args))

    w_init_var = lambda *args: (0.01 * jax.nn.initializers.glorot_normal()(*args))
    b_init_var = lambda *args: (jax.nn.initializers.zeros(*args) - 2)

    output_dim = nn_util.vectorize_pytree(dummy_output).shape[0]

    trunk_fn = None  # MLP(features=(3, 4, 5), kernel_init=w_init)
    mean_fn = nn.Dense(output_dim, kernel_init=w_init_mean)
    var_fn = nn.Dense(output_dim, kernel_init=w_init_var, bias_init=b_init_var)

    return independent_gaussian_proposal(n_proposals, stock_proposal_input_without_q_state, dummy_output,
                                         trunk_fn=trunk_fn, head_mean_fn=mean_fn,
                                         head_log_var_fn=var_fn)


def wrap_proposal_structure(PROPOSAL_STRUCTURE, proposal, param_vals):
    """
    Function that produces another function that wraps the proposal.  This needs wrapping because we define a
    proposal as a function that takes just the inputs (as opposed to the inputs and the parameters of the proposal).
    Therefore, this function partly just closes over the value of the proposal parameters, returning a function with
    a call to these values baked in.

    The proposal may also be parameterized in a funny way, and so this function provides some flexibility in how the
    function is defined and used.

    This is partly to separate out the code as much as possible, but also because vmapping over distribution and
    model objects was proving to be a pain, so this allows you to vmap the proposal inside the proposal itself,
    and then get the results of that and use them as required (i.e. for implementing the ResQ proposal).

    NOTE - both of the proposal functions also return a None, as there is no q_state to pass along.

    :param PROPOSAL_STRUCTURE:      String indicating the type of proposal structure to use.
    :param proposal:                Proposal object.  Will wrap a call to the `.apply` method.
    :param param_vals:              Tuple of parameters (p, q, r).  We will wrap q here.
    :return: Function that can be called as fn(inputs).
    """

    # If there is no proposal, then there is no structure to define.
    if proposal is None:
        return None

    # We fork depending on the proposal type.
    # Proposal takes arguments of (dataset, model, particles, time, p_dist, q_state, ...).
    if PROPOSAL_STRUCTURE == 'DIRECT':

        def _proposal(*_input):
            z_dist, q_state = proposal.apply(param_vals[1], _input)
            return z_dist, q_state

    elif PROPOSAL_STRUCTURE == 'RESQ':

        def _proposal(*_input):
            dataset, model, particles, t, p_dist, q_state = _input
            q_dist, q_state = proposal.apply(param_vals[1], _input)
            z_dist = tfd.MultivariateNormalFullCovariance(loc=p_dist.mean() + q_dist.mean(),
                                                          covariance_matrix=q_dist.covariance())
            return z_dist, q_state
    else:
        raise NotImplementedError()

    return _proposal



# TODO - BEGIN PAIN AND MISERY AND DEATH AND DESTRUCTION AND SADNESS.

def rebuild_model(_model, _param_vals, _p_params_accessors):
    for _v, _a in zip(_param_vals, _p_params_accessors):
        _model = _a(_model, _v)
    return _model

# TODO - END PAIN AND MISERY AND DEATH AND DESTRUCTION AND SADNESS.



def do_fivo_sweep(_key, _model, _proposal, _param_vals, _p_params_accessors, _dataset, _num_particles, _prop_structure):
    # Reconstruct the model, inscribing the new parameter values.
    _model = rebuild_model(_model, _param_vals[0], _p_params_accessors)

    # Reconstruct the proposal function.
    _proposal = wrap_proposal_structure(_prop_structure, _proposal, _param_vals)

    # Do the sweep.
    _smooth, _lmls, _, _ = smc(_key, _model, _dataset, proposal=_proposal, num_particles=_num_particles)

    # Compute the log of the expected marginal.
    _lml = lexp(_lmls)
    return - _lml, _smooth


def main():

    key = jr.PRNGKey(2)

    # Set up true model and draw some data.
    latent_dim = 3
    emissions_dim = 5
    num_trials = 5
    num_timesteps = 100
    num_particles = 100
    opt_steps = 100000

    # Create a more reasonable emission scale.
    transition_scale_tril = 0.1 * np.eye(latent_dim)
    emission_scale_tril = 0.5 * np.eye(emissions_dim)  # TODO - should be 1.0  .  0.1 for highly accurate observations.

    # Create the true model.
    key, subkey = jr.split(key)
    true_dynamics_weights = random_rotation(subkey, latent_dim, theta=np.pi / 10)
    true_model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey,
                             dynamics_scale_tril=transition_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_scale_tril=emission_scale_tril)

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)
    dummy_particles = true_model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2, ), )
    dummy_p_dist = true_model.dynamics_distribution(dummy_particles)

    # Now define a network to test.
    key, subkey = jax.random.split(key)
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey,
                        dynamics_scale_tril=transition_scale_tril,
                        emission_weights=true_model.emissions_matrix,
                        emission_scale_tril=emission_scale_tril)

    # Now define a proposal.
    PROPOSAL_STRUCTURE = 'RESQ'  # {'RESQ', 'DIRECT'}.
    proposal, proposal_params = None, None  # The default value for these is None.

    # Stock proposal input form is (dataset, model, particles, t, p_dist).
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((latent_dim, )), )
    proposal = define_proposal(n_proposals=1,
                               stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                               dummy_output=dummy_proposal_output)
    key, subkey = jr.split(key)
    proposal_params = proposal.init(subkey)


    # TODO - BEGIN PAIN AND MISERY AND DEATH AND DESTRUCTION AND SADNESS.
    def accessor_0(_model, _param=None):
        if _param is None:
            return _model._dynamics._distribution.weights
        else:
            _model._dynamics._distribution = ssmd.GaussianLinearRegression(_param,
                                                                           _model._dynamics._distribution.bias,
                                                                           _model._dynamics._distribution.scale_tril)
            return _model

    p_params_accessors = (accessor_0, )
    p_params = tuple(_a(model) for _a in p_params_accessors)
    # TODO - END PAIN AND MISERY AND DEATH AND DESTRUCTION AND SADNESS.

    # Build up the optimizer.
    opt = define_optimizer(p_params, proposal_params)

    with possibly_disabled():

        # Close over constant parameters.
        do_fivo_sweep_closed = lambda _k, _p, _num_particles: \
            do_fivo_sweep(_k, model, proposal, _p, p_params_accessors, dataset, _num_particles, PROPOSAL_STRUCTURE)

        # Jit this badboy.
        do_fivo_sweep_jitted = jax.jit(do_fivo_sweep_closed, static_argnums=2)

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        # Test the initial models.
        true_lml, em_log_marginal_likelihood, sweep_fig = \
            initial_validation(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted)

        for _step in range(opt_steps):

            key, subkey = jr.split(key)
            (pred_lml, smooth), grad = do_fivo_sweep_val_and_grad(subkey, get_params(opt), num_particles)
            opt = apply_gradient(grad, opt, )

            coldstart = 10
            if (_step % 1000 == 0) or (_step < coldstart):
                key, subkey = jr.split(key)
                do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood)

                if _step > coldstart:
                    pred_lml, pred_sweep = do_fivo_sweep_jitted(subkey, get_params(opt), _num_particles=5000)
                    sweep_fig = plot_single_sweep(pred_sweep[0], true_states[0], tag='{} Smoothing.'.format(_step),
                                                  fig=sweep_fig)

            p = 0

    print('Done')


if __name__ == '__main__':
    main()


