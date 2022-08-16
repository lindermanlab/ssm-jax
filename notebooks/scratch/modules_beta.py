from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as np
from jax import random as jr

from jax.tree_util import register_pytree_node_class
import jax

import copy

from contextlib import contextmanager

@register_pytree_node_class
class Parameter:
    def __init__(self, value, is_hyperparameter=False):
        self.is_hyperparameter=is_hyperparameter  # requires_grad
        self.value = value
        
    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.is_hyperparameter
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        is_hyperparameter = aux_data
        value, = children
        return cls(value, is_hyperparameter)
    
    def __repr__(self):
        return f"<Parameter {repr(self.value)} is_hyperparameter={self.is_hyperparameter}>"
        
@register_pytree_node_class
class Module:
    def __init__(self):
        self._parameters = dict()
        self._hyperparameters = dict()
        
    @property
    def parameters(self):
        return freeze(self._parameters)
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value.unfreeze()
        
    @property
    def hyperparameters(self):
        return freeze(self._hyperparameters)
    
    @hyperparameters.setter
    def hyperparameters(self, value):
        self._parameters = value.unfreeze()
    
    def tree_flatten(self):
        children = (self.parameters, self.hyperparameters)
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        parameters, hyperparameters = children
        return cls.from_parameters(parameters, hyperparameters)
    
    @contextmanager
    def inject(self, new_parameters):
        old_parameters = copy.deepcopy(self.parameters)
        self.parameters = new_parameters
        yield self
        self.parameters = old_parameters
    
    def __setattr__(self, name, value):
        
        # register module recursively
        if isinstance(value, Module):
            self._parameters[name] = value._parameters
            self._hyperparameters[name] = value._hyperparameters
        
        # register parameter / hyperparameter
        if isinstance(value, Parameter):
            
            if value.is_hyperparameter:
                self._hyperparameters[name] = value
            else:
                self._parameters[name] = value
                
        else:
            object.__setattr__(self, name, value)
            
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_hyperparameters' in self.__dict__:
            _hyperparameters = self.__dict__['_hyperparameters']
            if name in _hyperparameters:
                return _hyperparameters[name]
        if name in self.__dict__:
            return object.__getattr__(self, name)
        else:
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__.__name__, name))