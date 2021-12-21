from collections import namedtuple
import jax
import pytest

import jax.numpy as np

import ssm.utils as utils


def test_array_tree_equal():
    tree1 = (
        np.array([[0, 100, 200], [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15], [2, 6, 11, 16]]),
            np.array([[2, 12, 14], [3, 13, 15]]),
        ),
        "a",
    )

    tree2 = (
        np.array([[0, 100, 200], [1, 101, 999]]),
        (
            np.array([[1, 5, 10, 15], [2, 6, 11, 16]]),
            np.array([[2, 12, 14], [3, 13, 15]]),
        ),
        "a",
    )

    tree3 = (
        np.array([[0, 100, 200], [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15], [2, 6, 11, 16]]),
            np.array([[2, 12, 14], [3, 13, 15]]),
        ),
        "b",
    )

    assert utils.tree_all_equal(tree1, tree1)
    assert not utils.tree_all_equal(tree1, tree2)
    assert not utils.tree_all_equal(tree1, tree3)


def test_tree_get():

    tree = (
        np.array([[0, 100, 200], [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15], [2, 6, 11, 16]]),
            np.array([[2, 12, 14], [3, 13, 15]]),
        ),
    )

    first_batch_tree = (
        np.array([0, 100, 200]),
        (np.array([1, 5, 10, 15]), np.array([2, 12, 14])),
    )

    second_batch_tree = (
        np.array([1, 101, 201]),
        (np.array([2, 6, 11, 16]), np.array([3, 13, 15])),
    )

    idx_tree = utils.tree_get(tree, 0)
    assert utils.tree_all_equal(first_batch_tree, idx_tree)
    idx_tree = utils.tree_get(tree, 1)
    assert utils.tree_all_equal(second_batch_tree, idx_tree)
    
def test_tree_concatenate():
    tree1 = (
        np.array([[0, 100, 200], [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15], [2, 6, 11, 16]]),
            np.array([[2, 12, 14], [3, 13, 15]]),
        ),
    )

    tree_expected_axis_0 = (
        np.array([[0, 100, 200], [1, 101, 201],
                  [0, 100, 200], [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15], [2, 6, 11, 16],
                      [1, 5, 10, 15], [2, 6, 11, 16]]),
            np.array([[2, 12, 14], [3, 13, 15],
                      [2, 12, 14], [3, 13, 15]]),
        ),
    )
    
    tree_expected_axis_1 = (
        np.array([[0, 100, 200, 0, 100, 200], 
                  [1, 101, 201, 1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15, 1, 5, 10, 15],
                      [2, 6, 11, 16, 2, 6, 11, 16]]),
            np.array([[2, 12, 14, 2, 12, 14], 
                      [3, 13, 15, 3, 13, 15]]),
        ),
    )
    
    out = utils.tree_concatenate(tree1, tree1, axis=0)
    assert utils.tree_all_equal(tree_expected_axis_0, out)
    out = utils.tree_concatenate(tree1, tree1, axis=1)
    assert utils.tree_all_equal(tree_expected_axis_1, out)
    

@pytest.mark.parametrize("emissions_shape", ((5,), (5,10)))
def test_auto_batch(emissions_shape):
    
    class DummyModel:
        def __init__(self, emissions_shape):
            self.emissions_shape = emissions_shape

    @utils.auto_batch(batched_args=("data", "y"), model_arg="model")
    def f(data, y, model):
        # (B), T, D ==> (B,) D
        return (data + y).sum(axis=0)

    batch_dim = 3
    num_timesteps = 5
    model = DummyModel(emissions_shape=emissions_shape)
    batched_data = np.ones((batch_dim, num_timesteps) + emissions_shape)
    batched_y = np.ones((batch_dim, num_timesteps) + emissions_shape)

    # select single trial ==> shouldn't vmap
    no_batched_res = f(batched_data[0], batched_y[0], model)
    assert no_batched_res.shape == emissions_shape

    # select multiple trials ==> should vmap along batch_dim
    batched_res = f(batched_data, batched_y, model)
    assert batched_res.shape == (batch_dim,) + emissions_shape
    
@pytest.mark.parametrize("emissions_shape", ((5,), (5,10)))
def test_ensure_has_batch_dim(emissions_shape):
    
    class DummyModel:
        def __init__(self, emissions_shape):
            self.emissions_shape = emissions_shape
    
    @utils.ensure_has_batch_dim(batched_args=("data", "y"), model_arg="model")
    def f(data, y, model):
        # assert we always have our batch dim
        assert data.ndim == 2 + len(model.emissions_shape)  # (B, T, D)
        # since we assert data, y are batched we explicitly vmap
        return jax.vmap(lambda x, y: (x + y).sum(axis=0))(data, y)
    
    batch_dim = 3
    num_timesteps = 5
    model = DummyModel(emissions_shape=emissions_shape)
    batched_data = np.ones((batch_dim, num_timesteps) + emissions_shape)
    batched_y = np.ones((batch_dim, num_timesteps) + emissions_shape)

    # unbatched inputs should still have batch dim upon return
    no_batched_res = f(batched_data[0], batched_y[0], model)
    assert no_batched_res.shape == (1,) + emissions_shape

    # batched inputs should be left untouched
    batched_res = f(batched_data, batched_y, model)
    assert batched_res.shape == (batch_dim,) + emissions_shape
        
