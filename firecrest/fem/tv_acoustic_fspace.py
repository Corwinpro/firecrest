from firecrest.fem.base_fspace import BaseFunctionSpace
from firecrest.fem.struct_templates import Space


class TVAcousticFunctionSpace(BaseFunctionSpace):
    """
    Template function spaces for Thermo-Viscous Acoustic FEM problem.
    Chosen function spaces are:
        Pressure(scalar, order = 2),
        Velocity(vector, order = 2),
        Temperature(scalar, order = 1).
    If space is_complex, we are dealing with complex function space, i.e.
    every subspace (pressure, velocity, temperature) has both real and
    complex counterparts.
    """

    def __init__(self, domain, order=2, is_complex=False):
        self.spaces = (
            Space(element_type="CG", order=order, dimension="scalar"),
            Space(element_type="CG", order=order, dimension="vector"),
            Space(element_type="CG", order=order - 1, dimension="scalar"),
        )
        self.is_complex = is_complex
        if self.is_complex:
            self.spaces *= 2
        super().__init__(domain, self.spaces)
        self.pressure_function_space = self.function_spaces.sub(0)
        self.velocity_function_space = self.function_spaces.sub(1)
        self.temperature_function_space = self.function_spaces.sub(2)

    @property
    def pressure_function_space(self):
        if self.is_complex:
            raise NotImplementedError
            return (self.function_spaces.sub(0), self.function_spaces.sub(3))
        else:
            return self.function_spaces.sub(0)

    @property
    def velocity_function_space(self):
        if self.is_complex:
            raise NotImplementedError
            return (self.function_spaces.sub(1), self.function_spaces.sub(4))
        else:
            return self.function_spaces.sub(1)

    @property
    def temperature_function_space(self):
        if self.is_complex:
            raise NotImplementedError
            return (self.function_spaces.sub(2), self.function_spaces.sub(5))
        else:
            return self.function_spaces.sub(2)
