import pytest
from jax.interpreters import xla
import jax.random as jr
import jax.numpy as np
from tests.inference import utils as test_utils
import ssm.utils as utils
import warnings


@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()


def _test_smc_runs(key,
                   _model_constructor,
                   _proposal_constructor=test_utils.construct_default_proposal,
                   _tag='NoneSet'):
    """
    Verify that SMC still runs.
    Args:
        _model_constructor:
        _proposal_constructor:
        _tag:

    Returns:

    """
    
    # Define the model.
    key, subkey = jr.split(key)
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
    em_log_marginal_likelihood, smc_posteriors = test_utils.single_smc_test(key, model, prop_fn, data, num_rounds,
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
        assert False, ("{}:  Failed: Sampling from distribution: {}".format(_tag, err))


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
        assert False, ("{}:  Failed: Log prob evaluation: {}.".format(_tag, err))

    log_probs = 'N/A'
    try:
        log_probs = smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey))
        assert np.all(log_probs != 0.0)
    except:
        warnings.warn(UserWarning(("{}:  Soft warning:  log prob of some particles were equal to one.  " +
                                   "Generally imples gross degeneracy or some kind of failure...  " +
                                   "\nLog probs: {}".format(_tag, log_probs)), ))

    assert np.all(np.isfinite(smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)))), \
        "{}:  Failed: Log prob of particles was infinite."

    assert smc_posteriors.log_prob(smc_posteriors.sample(seed=subkey)).shape == (num_rounds, num_trials), \
        "{}:  Failed: Log probability shape: {} should be {}.".format(_tag, str(smc_posteriors.log_normalizer.shape),
                                                                 str((num_rounds, num_trials)))


def _test_bpf_single_model(key, _model_constructor, _tag='NoneSet'):
    """
    Verify that BPF runs and produces results close to EM.
    """
    
    # Define the model.
    key, subkey = jr.split(key)
    model = _model_constructor(subkey)
    
    # Sample the data.
    num_rounds = 20
    num_trials = 10
    num_timesteps = 100
    num_particles = 5000
    key, subkey = jr.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Run a single test.
    em_log_marginal_likelihood, smc_posteriors = test_utils.single_smc_test(key, model, None, data, num_rounds,
                                                                            num_particles, )

    assert np.all(np.isclose(utils.lexp(smc_posteriors.log_normalizer),
                             em_log_marginal_likelihood,
                             rtol=1e-02, atol=1.0)), \
        ("{}:  Failed: SMC/BPF Log prob evaluations are not sufficiently close to that of EM: \n".format(_tag) +
         "EM:      " + str(['{:5.3f}'.format(_f) for _f in em_log_marginal_likelihood]).replace('\n', '') + "\n" +
         "SMC/BPF: " + str(['{:5.3f}'.format(_f) for _f in utils.lexp(smc_posteriors.log_normalizer)]).replace('\n', ''))


def test_smc_runs_lds():
    key = jr.PRNGKey(0)
    _test_smc_runs(key, test_utils.construct_lds, test_utils.construct_lds_proposal, 'LDS')


def test_smc_runs_biglds():
    key = jr.PRNGKey(1)
    _test_smc_runs(key, test_utils.construct_big_lds, test_utils.construct_lds_proposal, 'Big LDS')


def test_smc_runs_bhmm():
    key = jr.PRNGKey(2)
    _test_smc_runs(key, test_utils.construct_bhmm, test_utils.construct_default_proposal, 'BHMM')


@pytest.mark.slow
def test_bpf_single_model_lds():
    key = jr.PRNGKey(3)
    _test_bpf_single_model(key, test_utils.construct_lds, 'LDS')


# @pytest.mark.slow
# def test_bpf_single_model_biglds():
#     key = jr.PRNGKey(4)
#     # NOTE - the SMC approximations for big models will be poor.
#     _test_bpf_single_model(key, _construct_big_lds, 'Big LDS')


@pytest.mark.slow
def test_bpf_single_model():
    key = jr.PRNGKey(5)
    _test_bpf_single_model(key, test_utils.construct_bhmm, 'BHMM')


if __name__ == '__main__':
    print('Beginning SMC tests.')

    # test_smc_runs_lds()
    # test_smc_runs_biglds()
    # test_smc_runs_bhmm()
    # test_bpf_single_model_lds()
    test_bpf_single_model()

    # # There is no real reason to expect this test to pass...
    # test_bpf_single_model_biglds()

    print('Tests complete.')
