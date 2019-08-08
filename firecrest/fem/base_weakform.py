from abc import ABC, abstractmethod
import dolfin as dolf
from firecrest.misc.type_checker import (
    is_numeric_argument,
    is_dolfin_exp,
    is_numeric_tuple,
)


class BaseWeakForm(ABC):
    def __init__(self, domain, **kwargs):
        self.domain = domain
        self.geometric_dimension = self.domain.mesh.ufl_cell().geometric_dimension()
        self.I = dolf.Identity(self.geometric_dimension)
        self.dirichlet_bcs = []

    def _parse_dolf_expression(self, expression):
        """
        Parses an (int, float, dolf.Constant, dolf.Expression) expression to dolfin-compatible
        format. We use this for generating values for dolf.DirichletBC.
        """
        if is_dolfin_exp(expression):
            value = expression
        elif is_numeric_argument(expression) or is_numeric_tuple(expression):
            value = dolf.Constant(expression)
        else:
            try:
                expression = expression.eval()
                value = self._parse_dolf_expression(expression)
            except AttributeError:
                raise TypeError(
                    f"Invalid boundary condition value type for boundary expression {expression}. "
                    f"It must be a compatible numerical value or dolfin value, or implement eval() method."
                )
        return value
