import dolfin as dolf


class BaseFunctionSpace:
    """
    Base Function Space factory.
    """

    def __init__(self, domain, spaces):
        """
        :param domain: geometrical domain with dolfin mesh
        :param spaces: collection of Space objects with Finite Element type, polynomial order, space dimension
        """
        self.domain = domain
        self.spaces = spaces
        self._function_spaces = None

    def _generate_function_spaces(self, **kwargs):
        """
        Generates a dolfin FunctionSpace of either single Finite Element (if only one Space is provided)
        or Mixed Elements (if more the one Space provided).
        """
        elements = [
            self._generate_finite_element(space, **kwargs) for space in self.spaces
        ]
        if len(elements) == 1:
            return dolf.FunctionSpace(self.domain.mesh, elements[0], **kwargs)

        mixed_element = dolf.MixedElement(elements)
        return dolf.FunctionSpace(self.domain.mesh, mixed_element, **kwargs)

    def _generate_finite_element(self, space, **kwargs):
        """
        Generates Finite Element for given Space.
        """
        cell = self.domain.mesh.ufl_cell()
        element_type = space.element_type or "Lagrange"

        if space.dimension == 1 or space.dimension == 0 or space.dimension == "scalar":
            return dolf.FiniteElement(element_type, cell, space.order)
        elif space.dimension == "vector":
            dimension = cell.geometric_dimension()
            return dolf.VectorElement(element_type, cell, space.order, dimension)
        elif space.dimension >= 2:
            return dolf.VectorElement(element_type, cell, space.order, space.dimension)

    @property
    def function_spaces(self):
        if not self._function_spaces:
            self._function_spaces = self._generate_function_spaces()

        return self._function_spaces
