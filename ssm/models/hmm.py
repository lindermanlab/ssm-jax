from ssm.models import Model
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


class HiddenMarkovModel(Model):

    """
    NOTE: do not have state-based stuff here

    i.e. anything that changes as as result of jitted operations
    should NOT be stored here
    """

    @staticmethod
    def dynamics(state, params, covariates):
        return NotImplementedError

    @staticmethod
    def emissions(state, params, covariates):
        return NotImplementedError

    @staticmethod
    def m_step():
        """
        How much do we want to disentangle models from inference?

        Key point: for some models, we can do better than a generic
        SGD approach. It makes sense to implement those methods here
        so as to take advantage of object-oriented inheritance
        conveniences.
        """
        return NotImplementedError


class GaussianHMM(Model):

    # parameters:
    # - transition matrices for each state k
    # - mean, covariance for each state k

    # dynamics: returns categorical             (take in vector of states --> vector of dists)
    # emissions: return Gaussian distribution   (can this be vmapped: take in vector of states --> vector of dists)

    # log likelihoods (in EM)
    # for each state:
    #   get emissions distribution --> call log_prob
    #   get dynamics distribution --> call log_prob [together these form transition matrix]

    # perhaps distinguish between parameters that are going to be optimized vs static

class GaussianObservationsJax:
    """
    Wrapper for a collection of Gaussian observation parameters.
    """

    def __init__(self, num_states, data_dim):
        """
        Initialize a collection of observation parameters for a Gaussian HMM
        with `num_states` (i.e. K) discrete states and `data_dim` (i.e. D)
        dimensional observations.
        """
        self.num_states = num_states
        self.data_dim = data_dim
        self.means = jnp.zeros((num_states, data_dim))
        self.covs = jnp.tile(jnp.eye(data_dim), (num_states, 1, 1))

    @staticmethod
    def precompute_suff_stats(dataset):
        """
        Compute the sufficient statistics of the Gaussian distribution for each
        data dictionary in the dataset. This modifies the dataset in place.

        Parameters
        ----------
        dataset: a list of data dictionaries.

        Returns
        -------
        Nothing, but the dataset is updated in place to have a new `suff_stats`
            key, which contains a tuple of sufficient statistics.
        """
        for data in dataset:
            x = data["data"]
            data['suff_stats'] = (jnp.ones(len(x)),                 # 1
                                  jnp.einsum('ti,tj->tij', x, x),   # x_t x_t^T
                                  x,                                # x_t
                                  jnp.ones(len(x)))                 # 1

    @staticmethod
    def log_likelihoods(data, means, covs):
        """
        Parameters
        ----------
        data: a dictionary with multiple keys, including "data", the TxD array
            of observations for this mouse.
        means: KxD array of observation means
        covs: KxDxD array of observation covariances

        Returns
        -------
        log_likes: a TxK array of log likelihoods for each datapoint and
            discrete state.
        """
        x = data["data"]
        K, _ = means.shape
        T, _ = x.shape
        dist = tfp.distributions.MultivariateNormalFullCovariance(means, covs)

        return dist.log_prob(x[:, None, :])

    @staticmethod
    def M_step(stats):
        """
        Compute the Gaussian parameters give the expected sufficient statistics.

        Note: add a little bit (1e-4 * I) to the diagonal of each covariance
            matrix to ensure that the result is positive definite.

        Parameters
        ----------
        stats: a tuple of expected sufficient statistics

        Returns
        -------
        Nothing, but self.means and self.covs are updated in place.
        """
        Ns, psi1, psi2, psi3 = stats

        means = psi2 / psi3[:, None]
        data_dim = means.shape[-1]

        covs = (1 / Ns[:, None, None]) * (psi1 - jnp.einsum('ti,tj->tij', psi2, psi2) / psi3[:, None, None])
        covs += 1e-4 * jnp.eye(data_dim)

        return means, covs

