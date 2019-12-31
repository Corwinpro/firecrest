import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod


class OptimizationMixin(ABC):
    """
    The Caching Mixin class is used to store the intermediate optimization results (direct state, adjoint state)
    for a given control parameters vector. It requires the optimization solver to implement the following methods:
    - _objective_state: result of the direct computation
    - _objective : the value of the objective function for an objective state
    - _jacobian: returns a gradient vector consistent with the control
    """

    def __init__(self, *args, **kwargs):
        """
        :attr default_renormalization: The objective and the Jacobian are normalized by a constant,
        such that the optimization tool does not require a lower convergence tolerance.
        """
        super().__init__(*args, **kwargs)
        self.objective_cache = {}
        self.jacobian_cache = {}
        self.verbose = kwargs.get("verbose", False)
        self.optimization_method = kwargs.get("optimization_method", "L-BFGS-B")

        self.default_renormalization = 1.0e6

    def objective(self, control):

        key = tuple(control)
        if key not in self.objective_cache:
            self.objective_cache[key] = self._objective_state(control)
            if self.verbose:
                print("added key: {}".format(key[::5]))
        state = self.objective_cache[key]
        objective = self._objective(state)
        if self.verbose:
            print("energy is: {}".format(objective))
        return objective * self.default_renormalization

    def jacobian(self, control):
        key = tuple(control)

        if key not in self.objective_cache:
            _ = self.objective(control)
        state = self.objective_cache[key]

        if key not in self.jacobian_cache:
            self.jacobian_cache[key] = np.array(self._jacobian(state))

        return self.jacobian_cache[key] * self.default_renormalization

    def minimize(self, x0, bnds):
        res = minimize(
            self.objective,
            x0,
            method=self.optimization_method,
            jac=self.jacobian,
            bounds=bnds,
            options={"disp": True, "maxiter": 40, "ftol": 1.0e-8},
        )
        return res

    @abstractmethod
    def _objective_state(self, control):
        pass

    @abstractmethod
    def _objective(self, state):
        pass

    @abstractmethod
    def _jacobian(self, state):
        pass
