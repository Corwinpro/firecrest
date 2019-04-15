from abc import ABC, abstractmethod
import dolfin as dolf


class BaseSolver(ABC):
    def __init__(self, domain, **kwargs):
        self.domain = domain
        self.mesh = self.domain.mesh
        self.boundaries = self.domain.boundary_elements
        self.n = dolf.FacetNormal(self.mesh)

    def dirichlet_boundary_condition(self, function_space, boundary):
        return dolf.DirichletBC(
            function_space, boundary.bcond, self.domain.ds, boundary.surface_index
        )

    def ds(self, boundary_type=None, boundary=None):
        return self.domain.get_boundary_measure(boundary_type, boundary)

    def dx(self):
        return self.domain.dx

    @abstractmethod
    def solve(self):
        pass


class TestSolver(BaseSolver):
    def solve(self):
        pass
