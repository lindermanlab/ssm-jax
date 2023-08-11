from functools import partial

import jax.numpy as np
from jax import vmap
from jax.tree_util import tree_map


__CONJUGATE_PRIORS = dict()

def register_prior(distribution, prior):
    __CONJUGATE_PRIORS[distribution] = prior


def get_prior(distribution):
    return __CONJUGATE_PRIORS[distribution]


class ConjugatePrior:
    r"""Interface for a conjugate prior distribution.

    TODO: give more detail
    """
    @classmethod
    def from_natural_parameters(cls, natural_params):
        """
        Construct an instance of the prior distribution given
        its natural parameters
        """
        raise NotImplementedError

    @property
    def natural_parameters(self):
        """Return the natural parameters of the distribution.
        These become pseudo-observations of the sufficient statistics
        of the conjugate distribution.
        """
        raise NotImplementedError


class ExponentialFamilyDistribution:
    r"""An interface for exponential family distributions
    with the necessary functionality for MAP estimation.

    .. math:
        p(x) = h(x) \exp\{t(x)^\top \eta - A(\eta)}

    where

    :math:`h(x)` is the base measure
    :math:`t(x)` are sufficient statistics
    :math:`\eta` are natural parameters
    :math:`A(\eta)` is the log normalizer

    """
    @classmethod
    def from_params(cls, params, **kwargs):
        """Create an instance parameters of the distribution
        with given parameters (e.g. the mode of a posterior distribution
        on those parameters). This function might have to do some conversion,
        e.g. from variances to scales.
        """
        raise NotImplementedError

    @classmethod
    def from_sufficient_statistics(cls, statistics, **kwargs):
        """Create an instance of the distribution with given statistics.
        This function might have to do some conversion, e.g. from second moment
        to covariances.
        """
        raise NotImplementedError

    @staticmethod
    def sufficient_statistics(data, **kwargs):
        """
        Return the sufficient statistics for each datapoint in an array,
        This function should assume the leading dimensions give the batch
        size.
        """
        raise NotImplementedError

    @classmethod
    def compute_conditional_from_stats(cls, stats):
        return get_prior(cls).from_natural_parameters(stats)

    @classmethod
    def compute_conditional(cls, data, weights=None, prior=None):
        # Flatten the data and weights so we can vmap over them
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_data = flatten(data)

        # Collect sufficient statistics for each data point
        stats = vmap(cls.sufficient_statistics)(flat_data)

        # Sum the (weighted) sufficient statistics
        if weights is not None:
            flat_weights = flatten(weights)
            stats = tree_map(lambda x: np.einsum('nk,n...->k...', flat_weights, x), stats)
        else:
            stats = tree_map(partial(np.sum, axis=0), stats)

        # Add the natural parameters from the prior
        if prior is not None:
            stats = tree_map(np.add, stats, prior.natural_parameters)

        # Compute the conditional distribution given the stats
        return cls.compute_conditional_from_stats(stats)

    @classmethod
    def compute_maximum_likelihood(cls, data, weights=None):
        # Flatten the data and weights so we can vmap over them
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_data = flatten(data)

        # Collect sufficient statistics for each data point
        stats = vmap(cls.sufficient_statistics)(flat_data)

        # Sum the (weighted) sufficient statistics
        if weights is not None:
            flat_weights = flatten(weights)
            fxn = lambda x: np.einsum('nk,n...->k...', flat_weights, x) / flat_weights.sum(axis=0)
            stats = tree_map(fxn, stats)
        else:
            stats = tree_map(partial(np.mean, axis=0), stats)

        # TODO compute standard parameters for sufficient statistics
        return cls.from_sufficient_statistics(stats)