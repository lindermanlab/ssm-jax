from scipy.optimize import linear_sum_assignment
import jax.numpy as np
from enum import IntEnum
from tqdm.auto import trange


class Verbosity(IntEnum):
    OFF = 0
    QUIET = 1
    LOUD = 2
    DEBUG = 3


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


def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == "int32" and z2.dtype == "int32"
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.sum((z1[:, None] == np.arange(K1))[:, :, None] &
                     (z2[:, None] == np.arange(K2))[:, None, :],
                     axis=0)
    assert overlap.shape == (K1, K2)
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm
