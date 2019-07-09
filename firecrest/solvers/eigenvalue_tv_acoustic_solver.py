from firecrest.solvers.base_solver import EigenvalueSolver
from firecrest.fem.tv_acoustic_weakform import ComplexTVAcousticWeakForm
import dolfin as dolf


class EigenvalueTVAcousticSolver(EigenvalueSolver):
    """
    Eigenvalue solver for thermoviscous acoustic problem.
    The problem is a generalized linear eigenvalue problem AA*x = s*BB*x.
    """

    def __init__(self, domain, complex_shift=0 + 0j, **kwargs):
        super().__init__(domain, **kwargs)
        self.complex_shift = complex_shift
        self.forms = ComplexTVAcousticWeakForm(domain, **kwargs)
        self.set_solver_operators(self.lhs, self.rhs)

    @property
    def lhs(self):
        """
        Constructs the LHS matrix (spatial components), AA of the eigenvalue problem.
        """
        spatial_component = -self.forms.spatial_component()
        imag_shift_components = -dolf.Constant(self.complex_shift.imag) * (
            self.forms.temporal_component("real", "imag")
            - self.forms.temporal_component("imag", "real")
        )
        real_shift_components = (
            dolf.Constant(self.complex_shift.real) * self.forms.temporal_component()
        )
        AA = dolf.PETScMatrix()
        AA = dolf.assemble(
            spatial_component + imag_shift_components + real_shift_components, tensor=AA
        )
        for bc in self.forms.dirichlet_boundary_conditions():
            bc.apply(AA)

        return AA.mat()

    @property
    def rhs(self):
        """
        Constructs the RHS matrix (temporal components), BB of the eigenvalue problem.
        """
        temporal_component = self.forms.temporal_component()
        BB = dolf.PETScMatrix()
        BB = dolf.assemble(temporal_component, tensor=BB)
        for bc in self.forms.dirichlet_boundary_conditions():
            bc.zero(BB)

        return BB.mat()

    def restore_eigenfunction(self, index, verbose=True):
        """
        Recombine complex vector solution of the eigenvalue problem back to normal,
        which appeared after doubling the space of the problem.
        See appendix of my First Year Report.

        :param index: index of (eigenvalue, eigenmode) to return
        :param verbose: printing the eigenvalue in runtime
        :return: a tuple of (eigenvalue, real part of the eigenmode,imaginary part of the eigenmode)
        """
        ev, rx, ix = self.retrieve_eigenvalue(index)
        real_part = self._vec_to_func(rx)
        imag_part = self._vec_to_func(ix)
        if verbose:
            print(ev + self.complex_shift)

        real_part = real_part.split(True)
        imag_part = imag_part.split(True)
        mid = int(len(imag_part) / 2)
        for j in range(len(real_part)):
            if j < mid:
                real_part[j].vector()[:] -= imag_part[j + mid].vector()
            else:
                real_part[j].vector()[:] += imag_part[j - mid].vector()

        return ev, real_part[:mid], real_part[mid:]

    def _vec_to_func(self, vector, function_space=None):
        if function_space is None:
            function_space = self.forms.function_space
        dolf_function = dolf.Function(function_space)
        dolf_function.vector()[:] = vector
        return dolf_function
