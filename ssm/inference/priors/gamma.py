from tensorflow_probability.substrates import jax as tfp


def _gamma_from_stats(stats, counts):
    (alpha,) = stats
    return tfp.distributions.Gamma(alpha, rate=counts[:, None])


def _gamma_pseudo_obs_and_counts(gamma):
    return (gamma.concentration,), gamma.rate


def _poisson_from_params(params):
    return tfp.distributions.Independent(
        tfp.distributions.Poisson(rate=params), reinterpreted_batch_ndims=1
    )


def _poisson_suff_stats(data):
    return (data,)
