from firecrest.fem.tv_acoustic_weakform import ComplexTVAcousticWeakForm
from firecrest.solvers.base_solver import BaseSolver
from collections import OrderedDict
import dolfin as dolf
from petsc4py import PETSc


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

    def create_ksp_solver(self, bilinear_matrix):
        """Creates KSP object with mumps as a preconditioner.
        :param PETSc.Mat bilinear_matrix: the bilinear matrix form
        :return: ksp object
        :rtype: PETSc.KSP
        """

        ksp = PETSc.KSP().create()
        ksp.setOperators(bilinear_matrix)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        return ksp

    def solve_petsc(self):
        """
        Solves the linear problem using PETSc interface, and manipulates with PETSc matrices.

        :return: list[state vectors]
        """
        form = self.forms._rhs_forms(shift=self.frequency) + self.forms._lhs_forms()
        w = dolf.Function(self.forms.function_space)

        lhs_matrix = dolf.PETScMatrix()
        lhs_matrix = dolf.assemble(dolf.lhs(form), tensor=lhs_matrix)
        for bc in self.forms.dirichlet_boundary_conditions():
            bc.apply(lhs_matrix)
        lhs_matrix = lhs_matrix.mat()

        averaged_boundary_terms = self.forms.boundary_averaged_velocity()
        if averaged_boundary_terms:
            lhs_matrix.axpy(-1.0, averaged_boundary_terms)

        solver = self.create_ksp_solver(lhs_matrix)

        rhs_vector = dolf.assemble(dolf.rhs(form))
        for bc in self.forms.dirichlet_boundary_conditions():
            bc.apply(rhs_vector)

        solver.solve(
            dolf.as_backend_type(rhs_vector).vec(),
            dolf.as_backend_type(w.vector()).vec(),
        )
        state = w.split(True)
        return state
