from ssm.inference.core import hmm_expected_states
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np


def EM(model, data):
    """
    Generic components

    for some num iters:
        E step(model, parameters, data) ==> posterior                  [need to be model-aware]
        M step(model, parameters, data, posterior) ==> new parameters  [need to be model-aware]

    E step(model, parameters, data)
        dict(model --> model-specific E step)

    E step HMM
    """


def hmm_e_step(model, parameters, data):

    # TODO: vectorize
    transition_matrix = model.transitions(np.arange(model.num_states), parameters)

    # compute log_likes, initial_dist, transition_matrix
    # pass into hmm_expected_states to get posterior
    # return posterior


# inheritance
# GaussianHMM.m_step(...)
# MyGaussianHMM.m_step(...)

# implementation?
# m_step_wrapper(GaussianHMM, ...)
#   under the hood: dispatching these parameters to a specific function

# could we still have OOM (e.g. pass in model via closure -- see SVAE)


def hmm_m_step(model, parameters, data, posteriors):

    # idea: replicate julia type-based dispatch
    # issue: need to plumb control room for modified object
    if isinstance(model, GaussianHMMModel):
        hmm_m_step_gaussian()


def hmm_m_step_sgd(model, parameters, data, posterior):

    # generically, we could write objective function E[log_joint_p]
    # do gradient ascent w/r/t parameters
    # E[log_joint_p] is convex for exponential model
    # marginal ll (actual optimization) is not strictly convex

    # ideally, would be model aware
    # to select how to compute the suff stats for that particular distribution
    # and compute the new parameters

    # expected log joint
    # from SSM:master
    def _expected_log_joint(expectations):
        elbo = self.log_prior()
        for data, input, mask, tag, (expected_states, _, _) in zip(
            datas, inputs, masks, tags, expectations
        ):
            lls = self.log_likelihoods(data, input, mask, tag)
            elbo += np.sum(expected_states * lls)
        return elbo

    # define optimization target
    T = sum([data.shape[0] for data in datas])

    def _objective(params, itr):
        self.params = params
        obj = _expected_log_joint(expectations)
        return -obj / T

    self.params = optimizer(_objective, self.params, **kwargs)

    return new_parameters


"""
these are natural parameters of HMM as exp fam
"""


def hmm_e_step(initial_dist, transition_matrix, log_likes):
    ll, suff_stats = hmm_expected_states(
        jnp.log(initial_dist), jnp.log(transition_matrix), log_likes
    )
    Ez0, Ezzp1, Ez = suff_stats

    posterior = dict()
    posterior["marginal_ll"] = ll
    posterior["Ez0"] = Ez0
    posterior["expected_states"] = Ez
    posterior["Ezzp1"] = Ezzp1

    return posterior


def fit_hmm(
    train_dataset,
    test_dataset,
    initial_dist,
    transition_matrix,
    observations,
    seed=0,
    num_iters=50,
):
    """
    Fit a Hidden Markov Model (HMM) with expectation maximization (EM).

    Note: This is only a partial fit, as this method will treat the initial
    state distribution and the transition matrix as fixed!

    Parameters
    ----------
    train_dataset: a list of dictionary with multiple keys, including "data",
        the TxD array of observations for this mouse, and "suff_stats", the
        tuple of sufficient statistics.

    test_dataset: as above but only used for tracking the test log likelihood
        during training.

    initial_dist: a length-K vector giving the initial state distribution.

    transition_matrix: a K x K matrix whose rows sum to 1.

    observations: an Observations object with `log_likelihoods` and `M_step`
        functions.

    seed: random seed for initializing the algorithm.

    num_iters: number of EM iterations.

    Returns
    -------
    train_lls: array of likelihoods of training data over EM iterations
    test_lls: array of likelihoods of testing data over EM iterations
    posteriors: final list of posterior distributions for the training data
    test_posteriors: final list of posterior distributions for the test data
    """
    # Get some constants
    num_states = observations.num_states
    num_train = sum([len(data["data"]) for data in train_dataset])
    num_test = sum([len(data["data"]) for data in test_dataset])

    # Check the initial distribution and transition matrix
    assert (
        initial_dist.shape == (num_states,)
        and jnp.all(initial_dist >= 0)
        and jnp.isclose(initial_dist.sum(), 1.0)
    )
    assert (
        transition_matrix.shape == (num_states, num_states)
        and jnp.all(transition_matrix >= 0)
        and jnp.allclose(transition_matrix.sum(axis=1), 1.0)
    )

    # Initialize with a random posterior
    posteriors = initialize_posteriors(train_dataset, num_states, seed=seed)
    stats = compute_expected_suff_stats(train_dataset, posteriors)

    # Track the marginal log likelihood of the train and test data
    train_lls = []
    test_lls = []

    def _train_step(stats, train_dataset, initial_dist, transition_matrix):
        # M step: update the parameters of the observations using the
        #         expected sufficient stats.
        means, covs = observations.M_step(stats)

        # E step: compute the posterior for each data dictionary in the dataset
        new_posteriors = []
        for data in train_dataset:
            log_likes = observations.log_likelihoods(data, means, covs)
            new_posterior = E_step_jax(initial_dist, transition_matrix, log_likes)
            new_posteriors.append(new_posterior)

        posteriors = new_posteriors

        # Compute the expected sufficient statistics under the new posteriors
        stats = compute_expected_suff_stats_jax(train_dataset, posteriors)

        # Store the average train likelihood
        avg_train_ll = sum([p["marginal_ll"] for p in posteriors]) / num_train
        return stats, means, covs, avg_train_ll, posteriors

    def _test_step(means, covs, initial_dist, transition_matrix):
        # Compute the posteriors for the test dataset too

        new_test_posteriors = []

        for data in test_dataset:
            log_likes = observations.log_likelihoods(data, means, covs)
            new_posterior = E_step_jax(initial_dist, transition_matrix, log_likes)
            new_test_posteriors.append(new_posterior)
        test_posteriors = new_test_posteriors

        # Store the average test likelihood
        avg_test_ll = sum([p["marginal_ll"] for p in test_posteriors]) / num_test
        return avg_test_ll, test_posteriors

    @jax.jit
    def _step(stats, train_dataset, initial_dist, transition_matrix):
        stats, means, covs, avg_train_ll, posteriors = _train_step(
            stats, train_dataset, initial_dist, transition_matrix
        )
        avg_test_ll, test_posteriors = _test_step(
            means, covs, initial_dist, transition_matrix
        )
        return stats, avg_train_ll, avg_test_ll, posteriors, test_posteriors

    # Main loop
    for itr in tqdm(range(num_iters)):
        stats, avg_train_ll, avg_test_ll, posteriors, test_posteriors = _step(
            stats, train_dataset, initial_dist, transition_matrix
        )
        train_lls.append(avg_train_ll)
        test_lls.append(avg_test_ll)

    # convert lls to arrays
    train_lls = jnp.array(train_lls)
    test_lls = jnp.array(test_lls)
    return train_lls, test_lls, posteriors, test_posteriors
