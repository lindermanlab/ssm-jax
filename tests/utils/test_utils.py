from collections import namedtuple
import pytest

import jax.numpy as np

import ssm.utils as utils

def test_array_tree_equal():
    tree1 = (
        np.array([[0, 100, 200],
                  [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15],
                      [2, 6, 11, 16]]),
            np.array([[2, 12, 14],
                      [3, 13, 15]])
        ),
        "a"
    )
    
    tree2 = (
        np.array([[0, 100, 200],
                  [1, 101, 999]]),
        (
            np.array([[1, 5, 10, 15],
                      [2, 6, 11, 16]]),
            np.array([[2, 12, 14],
                      [3, 13, 15]])
        ),
        "a"
    )
    
    tree3 = (
        np.array([[0, 100, 200],
                  [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15],
                      [2, 6, 11, 16]]),
            np.array([[2, 12, 14],
                      [3, 13, 15]])
        ),
        "b"
    )
    
    assert utils.tree_all_equal(tree1, tree1)
    assert not utils.tree_all_equal(tree1, tree2)
    assert not utils.tree_all_equal(tree1, tree3)

def test_tree_get():
    
    tree = (
        np.array([[0, 100, 200],
                  [1, 101, 201]]),
        (
            np.array([[1, 5, 10, 15],
                      [2, 6, 11, 16]]),
            np.array([[2, 12, 14],
                      [3, 13, 15]])
        )
    )
    
    first_batch_tree = (
        np.array([0, 100, 200]),
        (
            np.array([1, 5, 10, 15]),
            np.array([2, 12, 14])
        )
    )
    
    second_batch_tree = (
        np.array([1, 101, 201]),
        (
            np.array([2, 6, 11, 16]),
            np.array([3, 13, 15])
        )
    )
    
    idx_tree = utils.tree_get(tree, 0)
    assert utils.tree_all_equal(first_batch_tree, idx_tree)
    idx_tree = utils.tree_get(tree, 1)
    assert utils.tree_all_equal(second_batch_tree, idx_tree)