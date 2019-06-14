from firecrest.fem.base_fspace import BaseFunctionSpace
from firecrest.fem.struct_templates import Space


class TVAcousticFunctionSpace(BaseFunctionSpace):
    """
    Template function spaces for Thermo-Viscous Acoustic FEM problem.
    Chosen function spaces are:
        Pressure(scalar, order = 2),
        Velocity(vector, order = 2),
        Temperature(scalar, order = 1).
    """

    def __init__(self, domain, order=2):
        self.spaces = (
            Space(element_type="CG", order=order, dimension="scalar"),
            Space(element_type="CG", order=order, dimension="vector"),
            Space(element_type="CG", order=order - 1, dimension="scalar"),
        )
        super().__init__(domain, self.spaces)

    @property
    def pressure_function_space(self):
        """
        Picks pressure function space from generated function_spaces.
        """
        return self.function_spaces.sub(0)

    @property
    def velocity_function_space(self):
        """
        Picks velocity function space from generated function_spaces.
        """
        return self.function_spaces.sub(1)

    @property
    def temperature_function_space(self):
        """
        Picks temperature function space from generated function_spaces.
        """
        return self.function_spaces.sub(2)
