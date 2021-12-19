from slepc4py import SLEPc

from firecrest.solvers.base_solver import EigenvalueSolver
from firecrest.fem.tv_acoustic_weakform import ComplexTVAcousticWeakForm
import dolfin as dolf
from collections import OrderedDict


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
        lhs_matrix = dolf.PETScMatrix()
        lhs_forms = -self.forms._lhs_forms() - self.forms._rhs_forms(
            shift=self.complex_shift
        )
        lhs_matrix = dolf.assemble(dolf.lhs(lhs_forms), tensor=lhs_matrix)
        for bc in self.forms.dirichlet_boundary_conditions(is_linearised=True):
            bc.apply(lhs_matrix)

        return lhs_matrix.mat()

    @property
    def rhs(self):
        """
        Constructs the RHS matrix (temporal components), BB of the eigenvalue problem.
        """
        rhs_matrix = dolf.PETScMatrix()
        rhs_matrix = dolf.assemble(self.forms._rhs_forms(), tensor=rhs_matrix)
        for bc in self.forms.dirichlet_boundary_conditions(is_linearised=True):
            bc.zero(rhs_matrix)

        return rhs_matrix.mat()

    def extract_solution(self, index, eigenvalue_tolerance=1.0e-8, verbose=True):
        """
        Instead of passing an actual index from range(1, nof_converged), we pass the pair index.
        Then, we calculate the norms of the each solution in this pair, compare them, and return the
        one with the highest norm.
        :param index: int, number of pair
        :param eigenvalue_tolerance: float, tolerance value for difference between the real parts
        of the true and ghost solutions
        :param verbose: bool
        :return: a tuple of (eigenvalue, real part of the eigenmode,imaginary part of the eigenmode)
        """
        first_ev, rx, ix = self.retrieve_eigenpair(index * 2)
        first_norm, first_real, first_imag = self.reconstruct_eigenpair(rx, ix)

        second_ev, rx, ix = self.retrieve_eigenpair(index * 2 + 1)
        second_norm, second_real, second_imag = self.reconstruct_eigenpair(rx, ix)

        if abs(first_ev.real - second_ev.real) > eigenvalue_tolerance:
            print("Warning, the pair seems to be from different solution pairs.")

        if first_norm > second_norm:
            solution = first_ev + self.complex_shift, first_real, first_imag
        else:
            solution = second_ev + self.complex_shift, second_real, second_imag

        if verbose:
            print(solution[0])

        return solution

    def reconstruct_eigenpair(self, rx, ix, verbose=True):
        """
        Recombine complex vector solution of the eigenvalue problem back to normal,
        which appeared after doubling the space of the problem.
        See appendix of my First Year Report.

        :param verbose: output solution norm to verify it is non zero
        :param rx: real part vector of the solution
        :param ix: imaginary part vector of the solution
        :return: a tuple of (eigenvalue, real part of the eigenmode,imaginary part of the eigenmode)
        """

        real_part = self._vec_to_func(rx)
        imag_part = self._vec_to_func(ix)

        real_part = real_part.split(True)
        imag_part = imag_part.split(True)

        mid = int(len(imag_part) / 2)
        for j in range(len(real_part)):
            if j < mid:
                real_part[j].vector()[:] -= imag_part[j + mid].vector()
            else:
                real_part[j].vector()[:] += imag_part[j - mid].vector()

        norm = self._solution_norm(self._split_func_to_vec(real_part))
        if verbose:
            print(norm)

        return norm, real_part[:mid], real_part[mid:]

    def _vec_to_func(self, vector, function_space=None):
        if function_space is None:
            function_space = self.forms.function_space
        return super()._vec_to_func(vector, function_space)

    def _split_func_to_vec(self, function):
        """
        Given a split object (after a function.split(), returns a unified vector representation
        :param function: tuple of functions after split
        :return: dolfin.vector with elements assigned according to function values
        """
        dolf_function = dolf.Function(self.forms.function_space)
        for i in range(len(function)):
            dolf.assign(dolf_function.sub(i), function[i])
        return dolf_function.vector()

    def _solution_norm(self, vector):
        empty_vector = dolf.Function(self.forms.function_space).vector()
        self.lhs.mult(vector.vec(), empty_vector.vec())
        return empty_vector.norm("linf")

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

    def solve(self, number_of_modes=None):
        if number_of_modes is None:
            number_of_modes = self.nof_modes_to_converge
        else:
            number_of_modes = number_of_modes * 2
        self.solver.setDimensions(number_of_modes, SLEPc.DECIDE)
        super().solve()

        results = []
        for i in range(int(self.nof_modes_converged / 2)):
            ev, real_mode, imag_mode = self.extract_solution(i)
            results.append((ev, real_mode, imag_mode))

        return results
