import pytest
from jax.interpreters import xla
import jax.random as jr
import jax.numpy as np
from tests.inference import utils as test_utils

SEED = jr.PRNGKey(0)


@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()


def _test_fivo_runs(key, _model_constructor, _proposal_constructor, _tag='NoneSet'):
    """
    Quick test to make sure that FIVO can take steps.
    Args:
        _model_constructor:
        _proposal_constructor:
        _tag:

    Returns:

    """
    try:
        test_utils.run_fivo(key,
                            _tag,
                            _model_constructor,
                            _proposal_constructor,
                            _n_opt_steps=2)
    except Exception as err:
        print("{}:  Failed:  ")
        assert False


def _test_fivo_converges(key, _model_constructor, _proposal_constructor, _verbose=False, _tag='NoneSet'):
    """
    Slower test making sure that FIVO converges as expected.
    Args:
        _model_constructor:
        _proposal_constructor:
        _tag:

    Returns:

    """
    try:
        true_params, em_nlml, pred_nlml, cur_params = \
            test_utils.run_fivo(key,
                                _tag,
                                _model_constructor,
                                _proposal_constructor,
                                _n_opt_steps=10000,
                                _verbose=_verbose)
    except Exception as err:
        print("{}:  Failed:  {}".format(_tag, err))
        assert False

    # Now test that the parameters were close.
    learned_model_params = cur_params[0]
    for _p_l, _p_t, _k in zip(learned_model_params, true_params, true_params._fields):
        assert np.all(np.isclose(_p_l, _p_t, rtol=0.1, atol=5e-2)), \
            ('{}: Soft Warning: Parameter {} did not converge:'.format(_tag, _k) +
             '\nTrue params: {}'.format(_p_t.flatten()).replace('\n', '') +
             '\nPred params: {}'.format(_p_l.flatten()).replace('\n', ''))

    assert np.isclose(em_nlml, pred_nlml, rtol=0.1, atol=2.0), \
        ("{}: Failed: nlmls are not close enough: " +
         "\nTrue nlml: {: >6.2f}".format(em_nlml) +
         "\nPred nlml: {: >6.2f}".format(pred_nlml))

    if _verbose:
        print('True params: {}'.format(true_params[0].flatten()).replace('\n', '') + "\n" +
              'Pred params: {}'.format(learned_model_params[0].flatten()).replace('\n', ''))


def test_fivo_lds():
    key = jr.PRNGKey(0)
    _test_fivo_runs(key,
                    test_utils.construct_lds,
                    test_utils.construct_lds_proposal,
                    _tag='LDS-fivo-runs')


@pytest.mark.slow
def test_fivo_converges(_verbose=False):
    key = jr.PRNGKey(1)
    _test_fivo_converges(key,
                         test_utils.construct_lds,
                         test_utils.construct_lds_proposal,
                         _verbose=_verbose,
                         _tag='LDS-fivo-conv')


if __name__ == '__main__':
    print('Beginning FIVO tests.')
    test_fivo_lds()
    test_fivo_converges(_verbose=True)
    print('Tests complete.')
