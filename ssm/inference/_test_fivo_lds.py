import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.lds.models import GaussianLDS
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts


def lds_define_test(subkey):

    proposal_structure = None  # 'RESQ'         # {None/'BOOTSTRAP', 'RESQ', 'DIRECT', }
    tilt_structure = None  # 'DIRECT'

    # Define the true model.
    key, subkey = jr.split(subkey)
    true_model, true_states, dataset = lds_define_true_model_and_data(subkey)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = lds_define_test_model(subkey, true_model)

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


def lds_define_test_model(subkey, true_model, ):
    """

    :param subkey:
    :param true_model:
    :return:
    """

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    free_parameters = ('dynamics_weights', )

    # Generate a model to use.
    default_model = GaussianLDS(num_latent_dims=true_model.latent_dim,
                                num_emission_dims=true_model.emissions_shape[0],
                                dynamics_scale_tril=true_model._dynamics.scale_tril,
                                emission_weights=true_model.emissions_matrix,
                                emission_scale_tril=true_model._emissions.scale_tril,
                                seed=subkey)

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, free_parameters)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def lds_define_tilt(subkey, model, dataset, tilt_structure):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """

    if tilt_structure is None:
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    raise NotImplementedError("TODO - the tilt is not set up for general T yet...")

    # Tilt functions take in (dataset, model, particles, t-1).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    stock_tilt_input = (dataset[-1], model, dummy_particles[0], 0)

    # Define the proposal itself.
    tilt = tilts.IndependentGaussianTilt(n_tilts=1, tilt_input=stock_tilt_input)
    tilt_params = tilt.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_tilt_fn = tilts.rebuild_tilt(tilt, tilt_structure)
    return tilt, tilt_params, rebuild_tilt_fn


def lds_define_proposal(subkey, model, dataset, proposal_structure):
    """

    :param subkey:
    :param model:
    :param dataset:
    :return:
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

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=1,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output)
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def lds_define_true_model_and_data(key):
    """

    :param key:
    :return:
    """
    latent_dim = 3
    emissions_dim = 5
    num_trials = 50
    num_timesteps = 100

    # Create a more reasonable emission scale.
    transition_scale_tril = 0.1 * np.eye(latent_dim)
    emission_scale_tril = 0.5 * np.eye(emissions_dim)

    # Create the true model.
    key, subkey = jr.split(key)
    true_dynamics_weights = random_rotation(subkey, latent_dim, theta=np.pi / 10)
    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=transition_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_scale_tril=emission_scale_tril)

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    return true_model, true_states, dataset


def lds_do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood=None):
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
    print('True: dynamics:  ', '  '.join(['{: >9.3f}'.format(_s) for _s in true_model.dynamics_matrix.flatten()]))
    print('Pred: dynamics:  ', '  '.join(['{: >9.3f}'.format(_s) for _s in opt[0].target[0].flatten()]))

    # if opt[1] is not None:
    #     print('True: log-var:   ', '  '.join(['{: >9.5f}'.format(_s) for _s in np.log(np.diagonal(true_model.dynamics_noise_covariance))]))
    #     print('Pred: q log-var: ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[1].target._asdict()['head_log_var_fn'].W.flatten()]))
    print()
