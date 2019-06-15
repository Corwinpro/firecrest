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

    def solve(self, initial_state, time_scheme="implicit_euler"):
        if time_scheme == "implicit_euler":
            form, bcs = self.implicit_euler(initial_state)
        elif time_scheme == "crank_nicolson":
            form, bcs = self.crank_nicolson(initial_state)
        else:
            raise NotImplementedError(
                f"Time discretization scheme {time_scheme} is not yet implemented."
            )
        w = dolf.Function(self.forms.function_space)
        dolf.solve(dolf.lhs(form) == dolf.rhs(form), w, bcs)
        return w
