from firecrest.fem.base_fspace import BaseFunctionSpace
from firecrest.fem.space_template import Space


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
        spaces = (
            Space(element_type="CG", order=order, dimension="scalar"),
            Space(element_type="CG", order=order, dimension="vector"),
            Space(element_type="CG", order=order - 1, dimension="scalar"),
        )
        self.is_complex = is_complex
        if self.is_complex:
            spaces *= 2
        super().__init__(domain, spaces)
