"""
Useful utility functions.
"""

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
from jax import vmap, lax
from jax.tree_util import tree_map, tree_structure, tree_leaves

import inspect
from enum import IntEnum
from tqdm.auto import trange
from scipy.optimize import linear_sum_assignment
from typing import Sequence, Optional
from functools import wraps, partial
import copy


class Verbosity(IntEnum):
    """
    Convenience alias class for Verbosity values.

    Currently, any value >= 1 corresponds to displaying progress bars
    for various function calls through JAX-SSM.

    - 0: ``OFF``
    - 1: ``QUIET``
    - 2: ``LOUD``
    - 3: ``DEBUG``
    """

    OFF = 0
    QUIET = 1
    LOUD = 2
    DEBUG = 3


def tree_get(tree, idx):
    """Idx the leaves of the PyTree.

    Args:
        tree ([type]): [description]
        idx ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tree_map(lambda x: x[idx], tree)

def tree_concatenate(tree1, tree2, axis=0):
    """Concatenate leaves of two pytrees along specified axis.

    Args:
        tree1 ([type]): [description]
        tree2 ([type]): [description]
        axis ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tree_map(lambda x, y: np.concatenate((x, y), axis=axis), tree1, tree2)


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


def compute_state_overlap(
    z1: Sequence[int],
    z2: Sequence[int],
    K1: Optional[int] = None,
    K2: Optional[int] = None,
):
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

    overlap = np.sum(
        (z1[:, None] == np.arange(K1))[:, :, None]
        & (z2[:, None] == np.arange(K2))[:, None, :],
        axis=0,
    )
    assert overlap.shape == (K1, K2)
    return overlap


def find_permutation(
    z1: Sequence[int],
    z2: Sequence[int],
    K1: Optional[int] = None,
    K2: Optional[int] = None,
):
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


def random_rotation(seed, n, theta=None):
    """Helper function to create a rotating linear system.

    Args:
        seed (jax.random.PRNGKey): JAX random seed.
        n (int): Dimension of the rotation matrix.
        theta (float, optional): If specified, this is the angle of the rotation, otherwise
            a random angle sampled from a standard Gaussian scaled by ::math::`\pi / 2`. Defaults to None.

    Returns:
        [type]: [description]
    """

    key1, key2 = jr.split(seed)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * jr.uniform(key1)

    if n == 1:
        return jr.uniform(key1) * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out = out.at[:2, :2].set(rot)
    q = np.linalg.qr(jr.uniform(key2, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)


def ensure_has_batch_dim(batched_args=("data", "posterior", "covariates", "metadata"),
                         model_arg="self"):
    def ensure_has_batch_dim_decorator(f):
        sig = inspect.signature(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            assert "data" in batched_args and "data" in bound_args.arguments,\
                "`data` must be an argument in order to use the `ensure_has_batch_dim` decorator."

            assert model_arg in bound_args.arguments, \
                "`model_arg` must be an argument in order to use the `auto_batch` decorator."

            # Determine the batch dimension from the shape of the data and the model.
            # Naively assume that if data needs a batch, so do the other batched args.
            given_shape = tree_map(lambda x: np.array(x.shape), bound_args.arguments["data"])
            emissions_shape = bound_args.arguments[model_arg].emissions_shape

            # First check that the trailing shapes are correct.
            # leaf_is_valid = lambda shp, shp_suffix: \
            #     len(shp) > len(shp_suffix) \
            #         and len(shp) <=  len(shp_suffix) + 2 and \
            #             np.all(shp[-len(shp_suffix):] == shp_suffix)
            # assert all(tree_leaves(tree_map(leaf_is_valid, given_shape, emissions_shape)))

            # A leaf needs a batch dim if it only has one extra dimension (num_timesteps).
            leaf_needs_batch = lambda shp, shp_suffix: len(shp) == len(shp_suffix) + 1
            needs_batch = all(tree_leaves(tree_map(leaf_needs_batch, given_shape, emissions_shape)))

            # If data needs a batch, assume the other args do too
            if needs_batch:
                for key in batched_args:
                    if key in bound_args.arguments and bound_args.arguments[key] is not None:
                        bound_args.arguments[key] = \
                            tree_map(lambda x: x[None, ...], bound_args.arguments[key])

            return f(**bound_args.arguments)

        return wrapper
    return ensure_has_batch_dim_decorator


def auto_batch(batched_args=("data", "posterior", "covariates", "metadata", "states"),
               model_arg="self", map_function=vmap):
    def auto_batch_decorator(f):
        sig = inspect.signature(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            assert "data" in batched_args and "data" in bound_args.arguments,\
                "`data` must be an argument in order to use the `auto_batch` decorator."

            assert model_arg in bound_args.arguments, \
                "`model_arg` must be an argument in order to use the `auto_batch` decorator."

            # Determine the batch dimension from the shape of the data and the model.
            # Naively assume that if data is batched, so are the other batched args.
            given_shape = tree_map(lambda x: np.array(x.shape), bound_args.arguments["data"])
            emissions_shape = bound_args.arguments[model_arg].emissions_shape

            # First check that the trailing shapes are correct.
            # leaf_is_valid = lambda shp, shp_suffix: \
            #     len(shp) > len(shp_suffix) \
            #         and len(shp) <=  len(shp_suffix) + 2 and \
            #             np.all(shp[-len(shp_suffix):] == shp_suffix)
            # assert all(tree_leaves(tree_map(leaf_is_valid, given_shape, emissions_shape)))

            # Assume a leaf is batched if it has two extra dimensions (batch and num_timesteps).
            leaf_is_batched = lambda shp, shp_suffix: len(shp) == len(shp_suffix) + 2
            is_batched = all(tree_leaves(tree_map(leaf_is_batched, given_shape, emissions_shape)))

            if not is_batched:
                return f(*args, **kwargs)

            else:
                # Otherwise, separate out the fixed args from those with a batch dimension and call vmap.
                fixed_kwargs, batch_kwargs = {}, {}
                for arg, val in bound_args.arguments.items():
                    if arg in batched_args:
                        batch_kwargs[arg] = val
                    else:
                        fixed_kwargs[arg] = val
                return map_function(partial(f, **fixed_kwargs))(**batch_kwargs)

        return wrapper
    return auto_batch_decorator


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh = zoh.at[np.arange(N), np.arange(K)[np.ravel(z)]].set(1)
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def logspace_tensordot(tensor, matrix, axis):
    """
    Parameters
    ----------
    tensor : (..., m, ...)-array
    matrix : (m, n)-array
    axis : int

    Returns
    -------
    result : (..., n, ...)-array
    """
    tensor = np.moveaxis(tensor, axis, -1)
    tensor = spsp.logsumexp(tensor[..., None] + matrix, axis=-2)
    return np.moveaxis(tensor, -1, axis)


#### FUNCTIONS FOR DEBUGGING ####
# terminal color macros
CRED = "\033[91m"
CEND = "\033[0m"


def test_and_find_inequality(obj_a, obj_b, check_name="shape", mode="input", sig=None):
    """Iterates through zipped components of obj_a and obj_b to find inequality.

    Prints a message and returns the indices of unequal components in obj_a and obj_b.

    Args:
        obj_a ([type]): [description]
        obj_b ([type]): [description]
        check_name (str, optional): [description]. Defaults to "shape".
        mode (str, optional): [description]. Defaults to "input".
        sig ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    inequality_idxs = []
    if obj_a != obj_b:
        for i, (a, b) in enumerate(zip(obj_a, obj_b)):
            if a != b:
                print(
                    f"{CRED}[[{check_name} mismatch found for {mode} at index {i}"
                    f"{f' (arg={sig.args[i]})' if sig is not None else ''}]]"
                )
                print(f"prev={a}\ncurr={b}", CEND)
                inequality_idxs.append(i)
    return inequality_idxs




def check_pytree_structure_match(obj_a, obj_b, mode="input", sig=None):
    """Checks whether pytrees A and B have the same structure.
    Used for debugging re-jit problems (see debug_rejit decorator).

    Args:
        obj_a: pytree obj A (prev)
        obj_b: pytree obj B (curr)
        mode (str, optional): "input" or "output". Defaults to "input".
        sig (inspect.FullArgSpec, optional): optional function signature.
            Used for better debug description. Defaults to None.
    """
    struct_a = tree_structure(obj_a)
    struct_b = tree_structure(obj_b)
    idxs = test_and_find_inequality(
        struct_a.children(), struct_b.children(), check_name="PyTreeDef Structure", mode=mode, sig=sig
    )
    for i in idxs:
        print(f"{CRED}[{mode} pytree structure [{i}]]")
        print("prev=", repr(tree_leaves(obj_a)[i]))
        print("curr=", repr(tree_leaves(obj_b)[i]), CEND)


def check_pytree_shape_match(obj_a, obj_b, mode="input", sig=None):
    """Checks whether pytrees A and B have the same leaf shapes.
    Used for debugging re-jit problems (see debug_rejit decorator).

    Args:
        obj_a (jaxlib.xla_extension.PyTreeDef): pytree obj A (prev)
        obj_b (jaxlib.xla_extension.PyTreeDef): pytree obj B (curr)
        mode (str, optional): "input" or "output". Defaults to "input".
        sig (inspect.FullArgSpec, optional): doesn't support signature yet.
    """
    shape_a = [x.shape for x in tree_leaves(obj_a)]
    shape_b = [x.shape for x in tree_leaves(obj_b)]
    idxs = test_and_find_inequality(
        shape_a, shape_b, check_name="PyTree Leaf Shape", mode=mode, sig=None
    )
    for i in idxs:
        print(f"{CRED}[{mode} pytree leaf [{i}]]")
        print("prev=", repr(tree_leaves(obj_a)[i]))
        print("curr=", repr(tree_leaves(obj_b)[i]), CEND)

def check_pytree_weak_type_match(obj_a, obj_b, mode="input", sig=None):
    """Checks whether pytrees A and B have the same weak_typing.
    Used for debugging re-jit problems (see debug_rejit decorator).
    """
    shape_a = [x.weak_type for x in tree_leaves(obj_a)]
    shape_b = [x.weak_type for x in tree_leaves(obj_b)]
    idxs = test_and_find_inequality(
        shape_a, shape_b, check_name="Pytree Leaf Device Array Weak Type", mode=mode, sig=None
    )
    for i in idxs:
        print(f"{CRED}[{mode} pytree leaf [{i}]]")
        print("prev=", repr(tree_leaves(obj_a)[i]))
        print("curr=", repr(tree_leaves(obj_b)[i]), CEND)

def check_pytree_dtype_match(obj_a, obj_b, mode="input", sig=None):
    """Checks whether pytrees A and B have the same dtype.
    Used for debugging re-jit problems (see debug_rejit decorator).
    """
    shape_a = [x.dtype for x in tree_leaves(obj_a)]
    shape_b = [x.dtype for x in tree_leaves(obj_b)]
    idxs = test_and_find_inequality(
        shape_a, shape_b, check_name="Pytree Leaf Device Array dtype", mode=mode, sig=None
    )
    for i in idxs:
        print(f"{CRED}[{mode} pytree leaf [{i}]]")
        print("prev=", repr(tree_leaves(obj_a)[i]))
        print("curr=", repr(tree_leaves(obj_b)[i]), CEND)


def check_pytree_match(
    obj_a, obj_b, mode: str = "input", sig: inspect.FullArgSpec = None
):
    """Checks whether pytrees A and B are the same by checking shape, structure,
    weak_typing, and dtype.

    Used for debugging re-jit problems (see debug_rejit decorator).

    Args:
        obj_a (jaxlib.xla_extension.PyTreeDef): pytree structure A (prev)
        obj_b (jaxlib.xla_extension.PyTreeDef): pytree structure B (curr)
        mode (str, optional): "input" or "output". Defaults to "input".
        sig (inspect.FullArgSpec, optional): optional function signature.
            Used for better debug description. Defaults to None.
    """
    check_pytree_structure_match(obj_a, obj_b, mode, sig)
    check_pytree_shape_match(obj_a, obj_b, mode, None)
    check_pytree_weak_type_match(obj_a, obj_b, mode, None)
    check_pytree_dtype_match(obj_a, obj_b, mode, None)


def debug_rejit(func):
    """Decorator to debug re-jitting errors.

    Checks if input and output pytrees are consistent across multiple
    calls to func (else: func will need to be re-compiled).

    Example::

        @debug_rejit
        @jit
        def fn(inputs):
            return outputs

        # ==> will print out useful description when input/output
        #     pytrees mismatch (i.e. when fn will re-jit)
    """

    def wrapper(*args, **kwargs):

        # get tree structure for args and kwargs
        inputs = list(args) + list(kwargs.values())
        if wrapper.prev_in is None:
            wrapper.prev_in = inputs

        # run the function
        outputs = func(*args, **kwargs)

        # get tree structure for output (this works for tuple outputs too)
        if wrapper.prev_out is None:
            wrapper.prev_out = outputs

        # check whether the input and output structures match w/ prev fn call
        check_pytree_match(inputs, wrapper.prev_in, mode="input", sig=wrapper.sig)
        check_pytree_match(outputs, wrapper.prev_out, mode="output")

        # store for next fn call
        wrapper.prev_in = inputs
        wrapper.prev_out = outputs

        # return the output
        return outputs

    wrapper.sig = inspect.getfullargspec(func)
    wrapper.prev_in = None
    wrapper.prev_out = None
    return wrapper
