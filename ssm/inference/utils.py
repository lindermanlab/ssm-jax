from enum import IntEnum

import jax.numpy as jnp
from tqdm.auto import trange


def sum_tuples(a, b):
    assert a or b
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(ai + bi for ai, bi in zip(a, b))


def ssm_pbar(num_iters, verbose, description, *args):
    """Return either progress bar or regular list for iterating. Inputs are:
    num_iters (int)
    verbose (int)     - if == 2, return trange object, else returns list
    description (str) - description for progress bar
    args     - values to initialize description fields at
    """
    if verbose >= Verbosity.QUIET:
        pbar = trange(num_iters)
        pbar.set_description(description.format(*args))
    else:
        pbar = range(num_iters)
    return pbar


class Verbosity(IntEnum):
    OFF = 0
    QUIET = 1
    LOUD = 2
    DEBUG = 3


def compute_expected_suff_stats(dataset, posteriors):
    """
    Compute a tuple of expected sufficient statistics, taking a weighted sum
    of the posterior expected states and the sufficient statistics and combining.

    Parameters
    ----------
    dataset: a list of dictionary with multiple keys, including "data", the TxD
        array of observations, and "suff_stats", the tuple of sufficient statistics.

    Returns
    -------
    stats: a tuple of weighted sums of sufficient statistics. E.g. if the
        "suff_stats" key has four arrays, the stats tuple should have four
        entires as well. Each entry should be a K x (size of statistic) array
        with the expected sufficient statistics for each of the K discrete
        states.
    """
    assert isinstance(dataset, list)
    assert isinstance(posteriors, list)

    # Helper function to compute expected counts and sufficient statistics
    # for a single time series and corresponding posterior.
    def _compute_expected_suff_stats(data, posterior):

        q = posterior["expected_states"]
        stats = [jnp.einsum("tk,t...->k...", q, f) for f in data["suff_stats"]]
        stats = tuple(stats)

        return stats

    # Sum the expected stats over the whole dataset
    stats = None
    for data, posterior in zip(dataset, posteriors):
        these_stats = _compute_expected_suff_stats(data, posterior)
        stats = sum_tuples(stats, these_stats)
    return stats
