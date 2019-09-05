import warnings
import dolfin as dolf
from firecrest.misc.type_checker import (
    is_numeric_argument,
    is_dolfin_exp,
    is_numeric_tuple,
)


class BaseWeakForm:
    """
    `_complex_forms_flag` implements a special flag for complex valued boundary conditions.
    This allows us to switch between the real and complex values, and use the same forms
    generator for both the real, and the imaginary components of boundary conditions.
    """

    def __init__(self, domain, *args, **kwargs):
        self._domain = domain
        self.geometric_dimension = self.domain.mesh.ufl_cell().geometric_dimension()
        self.identity_tensor = dolf.Identity(self.geometric_dimension)
        self._complex_forms_flag = "real"

    @property
    def domain(self):
        return self._domain

    def _parse_real_dolf_expression(self, expression):
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

    @property
    def complex_forms_flag(self):
        return self._complex_forms_flag

    @complex_forms_flag.setter
    def complex_forms_flag(self, value):
        if value == "real" or value == "imag":
            pass
        else:
            raise ValueError("Only `real` or `imag` flag values accepted.")
        # warnings.warn(
        #     f"Changing the complex forms flag from {self.complex_forms_flag} to {value}"
        # )
        self._complex_forms_flag = value

    def _parse_dolf_expression(self, expression):
        if self.complex_forms_flag == "real":
            try:
                expression = expression.real
            except AttributeError:
                try:
                    expression = tuple(el.real for el in expression)
                except (AttributeError, TypeError):
                    pass
        elif self.complex_forms_flag == "imag":
            try:
                expression = expression.imag
            except AttributeError:
                expression = tuple(el.imag for el in expression)

        return self._parse_real_dolf_expression(expression)
