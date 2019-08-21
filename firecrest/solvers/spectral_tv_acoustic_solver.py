from firecrest.fem.tv_acoustic_weakform import ComplexTVAcousticWeakForm
from firecrest.solvers.base_solver import BaseSolver
from collections import OrderedDict
import dolfin as dolf


class SpectralTVAcousticSolver(BaseSolver):
    """
    Spectral solver for thermoviscous acoustic problem.
    The problem is a generalized linear problem omega*BB*x - AA*x = 0.
    """

    def __init__(self, domain, frequency=0 + 0j, **kwargs):
        super().__init__(domain, **kwargs)
        self.frequency = frequency
        self.forms = ComplexTVAcousticWeakForm(domain, **kwargs)

    def solve(self):
        form = self.forms._rhs_forms(shift=self.frequency) + self.forms._lhs_forms()
        w = dolf.Function(self.forms.function_space)
        dolf.solve(
            dolf.lhs(form) == dolf.rhs(form),
            w,
            self.forms.dirichlet_boundary_conditions(),
        )
        state = w.split(True)
        return state

    @property
    def visualization_files(self):
        if self._visualization_files is None:
            self._visualization_files = OrderedDict(
                {
                    "pR": dolf.File(self.vis_dir + "pressure_real.pvd"),
                    "uR": dolf.File(self.vis_dir + "u_real.pvd"),
                    "TR": dolf.File(self.vis_dir + "temperature_real.pvd"),
                    "pI": dolf.File(self.vis_dir + "pressure_imag.pvd"),
                    "uI": dolf.File(self.vis_dir + "u_imag.pvd"),
                    "TI": dolf.File(self.vis_dir + "temperature_imag.pvd"),
                }
            )
        return self._visualization_files
