from abc import ABC, abstractmethod
import dolfin as dolf


class BaseWeakForm(ABC):
    def __init__(self, domain, **kwargs):
        self.domain = domain
        self.geometric_dimension = self.domain.mesh.ufl_cell().geometric_dimension()
        self.I = dolf.Identity(self.geometric_dimension)
        self.dirichlet_bcs = []
