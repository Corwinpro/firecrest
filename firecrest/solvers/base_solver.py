from abc import ABC, abstractmethod
from slepc4py import SLEPc
from petsc4py import PETSc
import dolfin as dolf

LOG_LEVEL = 30


class BaseSolver(ABC):
    def __init__(self, domain, **kwargs):
        dolf.set_log_level(LOG_LEVEL)
        self.domain = domain

    @abstractmethod
    def solve(self):
        pass


class EigenvalueSolver(BaseSolver):
    """
    Base class for eigenvalue solver. It sets up the SLEPc solver.
    """

    def __init__(self, domain, **kwargs):
        super().__init__(domain)
        self.nof_modes_to_converge = kwargs.get("nmodes", 2)
        self.solver = self.configure_solver()

    def configure_solver(self):
        eps = SLEPc.EPS().create()
        st = eps.getST()
        st.setType("sinvert")

        # Set up the linear solver
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        # Set up number of (default) modes to converge
        eps.setDimensions(self.nof_modes_to_converge, SLEPc.DECIDE)

        return eps

    def set_solver_operators(self, AA, BB):
        """
        Set up eigensolver matrices, such that AA*x = s*BB*x.
        The matrices AA, BB are PETSc type objects.
        """
        self.AA = AA
        self.BB = BB
        self.solver.setOperators(self.AA, self.BB)

    def __solution_vector_template(self):
        rx = PETSc.Vec().createSeq(self.AA.getSize()[0])
        ix = PETSc.Vec().createSeq(self.AA.getSize()[0])
        return (rx, ix)

    def retrieve_eigenvalue(self, index):
        rx, ix = self.__solution_vector_template()
        eigenvalue = self.solver.getEigenpair(index, rx, ix)
        return eigenvalue, rx, ix

    def solve(self):
        """
        Solve the eigenvalue problem with configured solver and matrices
        """
        self.solver.solve()
        self.nof_modes_converged = self.solver.getConverged()
        print("Converged values:", self.nof_modes_converged)
