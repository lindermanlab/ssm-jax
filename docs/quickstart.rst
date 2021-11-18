Quickstart
==========

A quick little example of sampling from and fitting a GaussianHMM::

    from ssm.hmm import GaussianHMM
    import jax.random as jr

    # create a true HMM model
    hmm = GaussianHMM(num_states=5, num_emission_dims=10, seed=jr.PRNGKey(0))
    states, data = hmm.sample(key=jr.PRNGKey(1), num_steps=500, num_samples=5)

    # create a test HMM model
    test_hmm = GaussianHMM(num_states=5, num_emission_dims=10, seed=jr.PRNGKey(32))

    # fit it to our sampled data
    log_probs, fitted_model, posteriors = test_hmm.fit(data, method="em")

Check out these example notebooks to see ``SSM-JAX`` in action.

`All Notebooks on GitHub <https://github.com/lindermanlab/ssm-jax-refactor/tree/main/notebooks>`_

+---------+---------------------------------------------------------------------------------------------------------------------+
| Model   | Example(s)                                                                                                          |
+---------+---------------------------------------------------------------------------------------------------------------------+
| HMM     | - `Gaussian HMM <https://github.com/lindermanlab/ssm-jax-refactor/blob/main/notebooks/gaussian-hmm-example.ipynb>`_ |
|         | - `Poisson HMM <https://github.com/lindermanlab/ssm-jax-refactor/blob/main/notebooks/poisson-hmm-example.ipynb>`_   |
+---------+---------------------------------------------------------------------------------------------------------------------+
| ARHMM   | - `Gaussian ARHMM <https://github.com/lindermanlab/ssm-jax-refactor/blob/main/notebooks/arhmm-example.ipynb>`_      |
+---------+---------------------------------------------------------------------------------------------------------------------+
| LDS     | - `Gaussian LDS <https://github.com/lindermanlab/ssm-jax-refactor/blob/main/notebooks/gaussian-lds-example.ipynb>`_ |
|         | - `Poisson LDS <https://github.com/lindermanlab/ssm-jax-refactor/blob/main/notebooks/poisson-lds-example.ipynb>`_   |
+---------+---------------------------------------------------------------------------------------------------------------------+
