"""
Base model class.
"""


class Model:
    def __init__(self):
        """
        A 'model' is thought of as a collection of functions and a dictionary (parameters).
        The functions are implemented as static methods so as to enforce the functional
        paradigm of Jax.
        """
        self.parameters = None

    @staticmethod
    def dynamics(state, params, covariates):
        """
        Dynamics should return a probability distribution over the next state.

        dynamics(state, params, covariates) -> p(next_state)
        """
        raise NotImplementedError

    @staticmethod
    def emissions(state, params, covariates):
        """
        Emissions should return a probability distribution over the observations.

        emissions(state, params, covariates) -> p(observations)
        """
        raise NotImplementedError


