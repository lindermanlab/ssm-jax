import jax.numpy as jnp


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
