from firecrest.solvers.base_solver import BaseSolver
from firecrest.fem.tv_acoustic_weakform import TVAcousticWeakForm
import dolfin as dolf
from firecrest.misc.type_checker import (
    is_numeric_argument,
    is_numeric_tuple,
    is_dolfin_exp,
)
from collections import OrderedDict
import logging

DEFAULT_DT = 1.0e-3


class UnsteadyTVAcousticSolver(BaseSolver):
    def __init__(self, domain, **kwargs):
        super().__init__(domain)
        self.forms = TVAcousticWeakForm(domain, **kwargs)
        self._dt = kwargs.get("dt", DEFAULT_DT)
        self._inverse_dt = dolf.Constant(1.0 / self._dt)

        self.LUSolver = None
        self.bilinear_form = None

        self._initial_state = None

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state):
        state_list = []
        for component in state:
            if is_numeric_argument(component) or is_numeric_tuple(component):
                state_list.append(
                    dolf.Constant(component, cell=self.domain.mesh.ufl_cell())
                )
            elif is_dolfin_exp(component):
                state_list.append(component)
            else:
                raise TypeError(
                    "Numeric argument or dolf.Constant / dolf.Expression expected"
                )
        self._initial_state = tuple(state_list)

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
        self.initial_state = initial_state
        form, bcs = solving_scheme(self.initial_state)

        if self.LUSolver is None:
            self.initialize_solver(form, bcs)

        linear_form = dolf.rhs(form)
        res = dolf.assemble(linear_form)
        for bc in bcs:
            bc.apply(res)

        w = dolf.Function(self.forms.function_space)
        self.LUSolver.solve(self.bilinear_form, w.vector(), res)
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
        self.bilinear_form = dolf.assemble(dolf.lhs(form))
        for bc in bcs:
            bc.apply(self.bilinear_form)
        self.LUSolver = dolf.LUSolver(self.bilinear_form, solver_type)

    @property
    def visualization_files(self):
        if self._visualization_files is None:
            self._visualization_files = OrderedDict(
                {
                    "p": dolf.File(self.vis_dir + "pressure.pvd"),
                    "u": dolf.File(self.vis_dir + "u.pvd"),
                    "T": dolf.File(self.vis_dir + "temperature.pvd"),
                }
            )
        return self._visualization_files

    def output_field(self, fields, name=None):
        if name:
            fields.rename(name, name)
            self.visualization_files[name] << fields
        if len(fields) != len(self.visualization_files):
            raise IndexError(
                f"Expected {len(self.visualization_files)} fields, only {len(fields)} received."
            )
        for field, file_name in zip(fields, self.visualization_files):
            field.rename(file_name, file_name)
            self.visualization_files[file_name] << field
