"""
Useful utility functions.
"""

from scipy.optimize import linear_sum_assignment
import jax.numpy as np
from enum import IntEnum
from tqdm.auto import trange
from typing import Sequence, Optional


class Verbosity(IntEnum):
    """
    Convenience alias class for Verbosity values.

    - 0: ``OFF``
    - 1: ``QUIET`` 
    - 2: ``LOUD``
    - 3: ``DEBUG``
    """
    OFF = 0
    QUIET = 1
    LOUD = 2
    DEBUG = 3


def sum_tuples(a, b):
    """
    Utility function to sum tuples in an element-wise fashion.

    Args:
        a (tuple): A length ``n`` tuple
        b (tuple): A length ``n`` tuple

    Returns:
        c (tuple): The element-wise sum of ``a`` and ``b``. 
    """
    assert a or b
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(ai + bi for ai, bi in zip(a, b))


def ssm_pbar(num_iters, verbose, description, *args):
    """
    Return either progress bar or regular range for iterating depending on verbosity.

    Args:
        num_iters (int): The number of iterations for the iterator.
        verbose (int): if ``verbose == 2``, return ```trange`` object, else returns ``range``
        description (str): description for progress bar
        args: description format arguments
    """
    if verbose >= Verbosity.QUIET:
        pbar = trange(num_iters)
        pbar.set_description(description.format(*args))
    else:
        pbar = range(num_iters)
    return pbar


def compute_state_overlap(z1: Sequence[int], z2: Sequence[int], K1: Optional[int]=None, K2: Optional[int]=None):
    """
    Compute a matrix describing the state-wise overlap between two state vectors
    ``z1`` and ``z2``.

    The state vectors should both of shape ``(T,)`` and be integer typed.

    Args:
        z1: The first state vector.
        z2: The second state vector.
        K1: Optional upper bound of states to consider for ``z1``.
        K2: Optional upper bound of states to consider for ``z2``.

    Returns:
        overlap matrix: Matrix of cumulative overlap events. 
    """
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


def find_permutation(z1: Sequence[int], z2: Sequence[int], K1: Optional[int]=None, K2: Optional[int]=None):
    """
    Find the permutation between state vectors ``z1`` and ``z2`` that results in the most overlap.

    Useful for recovering the "true" state identities for a discrete-state SSM.

    Args:
        z1: The first state vector.
        z2: The second state vector.
        K1: Optional upper bound of states to consider for ``z1``.
        K2: Optional upper bound of states to consider for ``z2``.

    Returns:
        overlap matrix: Matrix of cumulative overlap events. 
    """
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm
