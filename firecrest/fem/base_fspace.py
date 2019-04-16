import dolfin as dolf


class BaseFunctionSpace:
    def __init__(self, domain, spaces):
        self.domain = domain
        self.spaces = spaces
        self.function_spaces = self.generate_function_spaces()

    def generate_function_spaces(self, **kwargs):
        elements = [
            self._generate_function_space(space, **kwargs) for space in self.spaces
        ]
        if len(elements) == 1:
            return dolf.FunctionSpace(self.domain.mesh, elements[0], **kwargs)

        mixed_element = dolf.MixedElement(elements)
        return dolf.FunctionSpace(self.domain.mesh, mixed_element, **kwargs)

    def _generate_function_space(self, space, **kwargs):
        cell = self.domain.mesh.ufl_cell()
        element_type = space.element_type or "Lagrange"
        return self.make_element(element_type, cell, space.order, space.dimension)

    @staticmethod
    def make_element(el_type, cell, order, dimension):
        if dimension == 1 or dimension == 0 or dimension == "scalar":
            return dolf.FiniteElement(el_type, cell, order)
        elif dimension == "vector":
            dimension = cell.geometric_dimension()
            return dolf.VectorElement(el_type, cell, order, dimension)
        elif dimension >= 2:
            return dolf.VectorElement(el_type, cell, order, dimension)
