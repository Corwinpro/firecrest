from abc import ABC, abstractmethod
import dolfin as dolf


class BaseWeakForm(ABC):
    def __init__(self, domain, **kwargs):
        self.domain = domain
        self.geometric_dimension = self.domain.mesh.ufl_cell().geometric_dimension()
        self.I = dolf.Identity(self.geometric_dimension)
        self.dirichlet_bcs = []

    @staticmethod
    def _parse_dolf_expression(expression):
        """
        Parses an (int, float, dolf.Constant, dolf.Expression) expression to dolfin-compatible
        format. We use this for generating values for dolf.DirichletBC.
        """
        if isinstance(expression, dolf.function.expression.Expression):
            value = expression
        elif isinstance(expression, (int, float)):
            value = dolf.Constant(expression)
        elif isinstance(expression, dolf.function.constant.Constant):
            value = expression
        else:
            raise TypeError(
                f"Invalid boundary condition value type for boundary expression {expression}."
            )
        return value
