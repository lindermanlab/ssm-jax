from flax.core.frozen_dict import FrozenDict
from contextlib import contextmanager
import copy

class Module:
    @contextmanager
    def inject(self, new_parameters: FrozenDict):
        old_parameters = copy.deepcopy(self._parameters)
        self._parameters = new_parameters
        yield self
        self._parameters = old_parameters
        
    @property
    def _parameters(self) -> FrozenDict:
        return FrozenDict()

    @_parameters.setter
    def _parameters(self, params):
        self._parameters = params

    @property
    def _hyperparameters(self) -> FrozenDict:
        return FrozenDict()

    @_hyperparameters.setter
    def _hyperparameters(self, hyperparams: FrozenDict):
        self._hyperparameters = hyperparams

    def tree_flatten(self):
        children = (self._parameters, self._hyperparameters)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, hyperparam = children
        obj = object.__new__(cls)
        obj._parameters = params
        obj._hyperparameters = hyperparam
        