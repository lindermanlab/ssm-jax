import pytest
import tensorflow_probability.substrates.jax as tfp
from jax.interpreters import xla
from jax import vmap
import jax
import jax.scipy.special as spsp
import jax.random as jr
import jax.numpy as np
import pytest

from ssm.hmm import BernoulliHMM
from ssm.lds import GaussianLDS
from ssm.inference.smc import smc
import ssm.utils as utils
import ssm.inference.proposals as proposals
import ssm.nn_util as nn_util
import flax.linen as nn


SEED = jr.PRNGKey(0)

# Allowable deviation of ML ratio from 1.0.
LML_RTOL = 0.1


@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()


def _single_smc_test(key, model, proposal, data, num_rounds, num_particles, ):
    """

    Args:
        key:
        model:
        proposal:
        data:
        num_rounds:
        num_particles:

    Returns:

    """

    # Test against EM (which for the LDS is exact.
    em_posterior = vmap(model.e_step)(data)
    em_log_marginal_likelihood = model.marginal_likelihood(data, posterior=em_posterior)

    # Close over a single SMC sweep.
    repeat_smc = lambda _k: smc(_k, model, len(data), data, proposal=proposal, num_particles=num_particles)
    vmapped = vmap(repeat_smc)

    # Run all the sweeps.
    key, subkey = jr.split(key)
    smc_posteriors = vmapped(jr.split(subkey, num=num_rounds))

    return em_log_marginal_likelihood, smc_posteriors


def _construct_bhmm(subkey):
    """

    Args:
        subkey:

    Returns:

    """
    latent_dim = 2
    emissions_dim = 3
    key, subkey = jr.split(subkey)
    model = BernoulliHMM(num_states=latent_dim, num_emission_dims=emissions_dim, seed=subkey)
    return model


def _construct_lds(subkey):
    """

    Args:
        subkey:

    Returns:

    """
    latent_dim = 2
    emissions_dim = 3
    model_kwargs = {'dynamics_scale_tril': 0.1 * np.eye(latent_dim),
                    'emission_scale_tril': 0.1 * np.eye(emissions_dim), }
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, **model_kwargs)
    return model


def _construct_big_lds(subkey):
    """

    Args:
        subkey:

    Returns:

    """
    latent_dim = 5
    emissions_dim = 10
    model_kwargs = {'dynamics_scale_tril': 0.05 * np.eye(latent_dim),
                    'emission_scale_tril': 0.2 * np.eye(emissions_dim), }
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, **model_kwargs)
    return model


def _construct_default_proposal(subkey, model, data):
    """
    We don't have proposals implemented for discrete models yet.
    Args:
        subkey:
        model:
        data:

    Returns:

    """
    return None, None, (lambda *args: None)


def _construct_lds_proposal(subkey, model, dataset):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """
    proposal_structure = 'RESQ'         # {None/'BOOTSTRAP', 'RESQ', 'DIRECT', }
    proposal_type = 'SHARED'            # {'SHARED', 'INDPENDENT'}

    if (proposal_structure is None) or (proposal_structure == 'BOOTSTRAP'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )
    output_dim = nn_util.vectorize_pytree(dummy_proposal_output).shape[0]

    w_init_mean = lambda *args: (0.1 * jax.nn.initializers.normal()(*args))
    head_mean_fn = nn.Dense(output_dim, kernel_init=w_init_mean)

    w_init_mean = lambda *args: ((0.1 * jax.nn.initializers.normal()(*args)) - 3)
    head_log_var_fn = nn_util.Static(output_dim, kernel_init=w_init_mean)

    # Define the number of proposals to define depending on the proposal type.
    if proposal_type == 'SHARED':
        n_proposals = 1
    else:
        n_proposals = len(dataset)

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=n_proposals,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def test_smc_runs(_tag, _model_constructor, _proposal_constructor=_construct_default_proposal):
    """
    Verify that SMC still runs.
    Args:
        _model_constructor:

    Returns:

    """
    
    # Define the model.
    key, subkey = jr.split(SEED)
    model = _model_constructor(subkey)
    
    # Sample the data.
    num_rounds = 5
    num_trials = 4
    num_timesteps = 10
    num_particles = 15
    key, subkey = jr.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Define the proposal to use.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = _proposal_constructor(subkey, model, data)
    prop_fn = rebuild_prop_fn(proposal_params)

    # Run a single test.
    em_log_marginal_likelihood, smc_posteriors = _single_smc_test(key, model, prop_fn, data, num_rounds,
                                                                  num_particles, )

    # Check the shapes.
    assert smc_posteriors.log_normalizer.shape == (num_rounds, num_trials), \
        "{}:  Failed: Log normalizer shape: {} should be {}.".format(_tag, str(smc_posteriors.log_normalizer.shape),
                                                                str((num_rounds, num_trials)))

    assert smc_posteriors.batch_shape == (num_rounds, ), \
        "{}:  Failed: Batch shape: {} should be {}.".format(_tag, str(smc_posteriors.batch_shape),
                                                       str((num_rounds, )))

    if smc_posteriors._has_state_dim:
        assert smc_posteriors.event_shape == (num_trials, num_particles, num_timesteps, model.latent_dim), \
            "{}:  Failed:(1) Event shape: {} should be {}.".format(_tag, str(smc_posteriors.event_shape),
                                                           str((num_trials, num_particles, num_timesteps, model.latent_dim)))
    else:
        assert smc_posteriors.event_shape == (num_trials, num_particles, num_timesteps), \
            "{}:  Failed:(2) Event shape: {} should be {}.".format(_tag, str(smc_posteriors.event_shape),
                                                              str((num_trials, num_particles, num_timesteps)))

    try:
        smc_posteriors.sample(seed=subkey)
    except Exception as err:
        print("{}:  Failed: Sampling from distribution: {}".format(_tag, err))
        assert False

    if smc_posteriors._has_state_dim:
        assert smc_posteriors.sample(seed=subkey).shape == (num_rounds, num_trials, num_timesteps, model.latent_dim), \
            "{}:  Failed:(1) Sample shape: {} should be {}.".format(_tag, str(smc_posteriors.event_shape),
                                                               str((num_rounds, num_trials, num_timesteps, model.latent_dim)))
    else:
        assert smc_posteriors.sample(seed=subkey).shape == (num_rounds, num_trials, num_timesteps), \
            "{}:  Failed:(2) Sample shape: {} should be {}.".format(_tag, str(smc_posteriors.event_shape),
                                                               str((num_rounds, num_trials, num_timesteps)))

    try:
        np.sum(smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)))
    except Exception as err:
        print("{}:  Failed: Log prob evaluation: {}.".format(_tag, err))
        assert False

    log_probs = 'N/A'
    try:
        log_probs = smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey))
        assert np.all(log_probs != 0.0)
    except:
        print("{}:  Soft warning:  log prob of some particles were equal to one.  Generally imples gross degeneracy"
              " or some kind of failure...  \nLog probs: {}".format(_tag, log_probs))
        assert False

    assert np.all(np.isfinite(smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)))), \
        "{}:  Failed: Log prob of particles was infinite."

    assert smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)).shape == (num_rounds, num_trials), \
        "{}:  Failed: Log probability shape: {} should be {}.".format(_tag, str(smc_posteriors.log_normalizer.shape),
                                                                 str((num_rounds, num_trials)))


@pytest.mark.slow
def test_bpf_single_model(_tag, _model_constructor):
    """
    Verify that BPF runs and produces results close to EM.
    """
    
    # Define the model.
    key, subkey = jr.split(SEED)
    model = _model_constructor(subkey)
    
    # Sample the data.
    num_rounds = 20
    num_trials = 10
    num_timesteps = 100
    num_particles = 5000
    key, subkey = jr.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Run a single test.
    em_log_marginal_likelihood, smc_posteriors = _single_smc_test(key, model, None, data, num_rounds,
                                                                  num_particles, )

    assert np.all(np.isclose(utils.lexp(smc_posteriors.log_normalizer),
                             em_log_marginal_likelihood,
                             rtol=1e-03, atol=0.5)), \
        ("{}:  Failed: SMC/BPF Log prob evaluations are not sufficiently close to that of EM: \n".format(_tag) +
         "EM:      " + str(['{:5.3f}'.format(_f) for _f in em_log_marginal_likelihood]).replace('\n', '') + "\n" +
         "SMC/BPF: " + str(['{:5.3f}'.format(_f) for _f in utils.lexp(smc_posteriors.log_normalizer)]).replace('\n', ''))


if __name__ == '__main__':
    print('Beginning SMC tests.')

    test_smc_runs('LDS', _construct_lds, _construct_lds_proposal)
    test_bpf_single_model('LDS', _construct_lds)

    test_smc_runs('Big LDS', _construct_big_lds, _construct_lds_proposal)
    # test_bpf_single_model('Big LDS', _construct_big_lds)  # NOTE - the SMC approximations for big models will be poor.

    test_smc_runs('BHMM', _construct_bhmm)
    test_bpf_single_model('BHMM', _construct_bhmm)

    print('Tests complete.')
