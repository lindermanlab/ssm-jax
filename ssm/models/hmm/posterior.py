"""
HMM Posterior Class
===================
"""
from collections import namedtuple

HMMPosterior = namedtuple(
    "HMMPosterior", ["marginal_likelihood", "expected_states", "expected_transitions"]
)
