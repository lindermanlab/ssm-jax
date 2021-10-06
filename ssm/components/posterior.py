"""
Prototype location for posterior objects.
"""


# TODO @slinderman @schlagercollin: Consider data classes
from collections import namedtuple

# TODO: potentially a light wrapper over a posterior distribution?
class Posterior:
    def __init__(self):
        pass

HMMPosterior = namedtuple(
    "HMMPosterior", ["marginal_likelihood", "expected_states", "expected_transitions"]
)