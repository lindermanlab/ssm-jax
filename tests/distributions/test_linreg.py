from typing import no_type_check


import pytest

import jax.numpy as np
import jax.random as jr

from ssm.distributions import GaussianLinearRegression


SEED = jr.PRNGKey(0)

def test_expected_log_prob(d_in=2, d_out=3):
    k1, k2, k3, k4 = jr.split(SEED, 4)

    W = jr.normal(k1, shape=(d_out, d_in))
    b = jr.normal(k2, shape=(d_out,))
    Q = np.eye(d_out)
    lr = GaussianLinearRegression(W, b, Q)

    x = jr.normal(k3, shape=(d_in,))
    y = jr.normal(k4, shape=(d_out,))

    lp_comp = lr.log_prob(y, covariates=x)
    lp = lr.expected_log_prob(y, x, np.outer(y, y), np.outer(y, x), np.outer(x, x))
    assert np.allclose(lp_comp, lp, atol=1e-5)