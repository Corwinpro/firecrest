from firecrest.solvers.base_solver import BaseSolver
from firecrest.fem.tv_acoustic_weakform import TVAcousticWeakForm
import dolfin as dolf
import logging

DEFAULT_DT = 1.0e-3


class UnsteadyTVAcousticSolver(BaseSolver):
    def __init__(self, domain, **kwargs):
        super().__init__(domain)
        self.forms = TVAcousticWeakForm(domain, **kwargs)
        self._dt = kwargs.get("dt", DEFAULT_DT)
        self._inverse_dt = dolf.Constant(1.0 / self._dt)

        self.LUSolver = None

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state

    def implicit_euler(self, initial_state):
        temporal_component = self.forms.temporal_component()
        temporal_component_old = self.forms.temporal_component(initial_state)

        spatial_component = self.forms.spatial_component()
        stress_boundary_component, temperature_boundary_component = (
            self.forms.boundary_components()
        )
        dirichlet_bcs = self.forms.dirichlet_boundary_conditions()

        form = (
            self._inverse_dt * (temporal_component - temporal_component_old)
            + spatial_component
            + stress_boundary_component
            + temperature_boundary_component
        )
        return form, dirichlet_bcs

    def crank_nicolson(self, initial_state):
        temporal_component = self.forms.temporal_component()
        temporal_component_old = self.forms.temporal_component(initial_state)

        spatial_component = self.forms.spatial_component()
        stress_boundary_component, temperature_boundary_component = (
            self.forms.boundary_components()
        )
        dirichlet_bcs = self.forms.dirichlet_boundary_conditions()
        spatial_component_old = self.forms.spatial_component(initial_state)
        stress_boundary_component_old, temperature_boundary_component_old = self.forms.boundary_components(
            initial_state
        )
        form = (
            self._inverse_dt * (temporal_component - temporal_component_old)
            + dolf.Constant(0.5) * (spatial_component + spatial_component_old)
            + dolf.Constant(0.5)
            * (stress_boundary_component + stress_boundary_component_old)
            + dolf.Constant(0.5)
            * (temperature_boundary_component + temperature_boundary_component_old)
        )
        return form, dirichlet_bcs

    def solve(self, initial_state, time_scheme="crank_nicolson"):
        try:
            solving_scheme = getattr(self, time_scheme)
        except AttributeError:
            raise NotImplementedError(
                f"Time discretization scheme {time_scheme} is not yet implemented."
            )

        form, bcs = solving_scheme(initial_state)

        if self.LUSolver is None:
            self.initialize_solver(form, bcs)

        L_form = dolf.rhs(form)
        res = dolf.assemble(L_form)
        for bc in bcs:
            bc.apply(res)

        w = dolf.Function(self.forms.function_space)
        self.LUSolver.solve(self.K, w.vector(), res)
        return w

    def initialize_solver(self, form, bcs, solver_type="mumps"):
        """
        Performs solver initialization and matrix factorization is stored.
        As discussed at https://fenicsproject.org/docs/dolfin/dev/python/demos/elastodynamics/demo_elastodynamics.py.html:
        'Since the system matrix to solve is the same for each time step (constant time step), 
        it is not necessary to factorize the system at each increment. It can be done once and 
        for all and only perform assembly of the varying right-hand side and backsubstitution 
        to obtain the solution much more efficiently. 
        This is done by defining a LUSolver object while PETSc handles caching factorizations.'
        """
        self.K = dolf.assemble(dolf.lhs(form))
        for bc in bcs:
            bc.apply(self.K)
        self.LUSolver = dolf.LUSolver(self.K, solver_type)
