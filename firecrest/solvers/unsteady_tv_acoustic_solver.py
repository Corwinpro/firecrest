from firecrest.solvers.base_solver import BaseSolver
from firecrest.fem.tv_acoustic_weakform import TVAcousticWeakForm
import dolfin as dolf
from firecrest.misc.type_checker import (
    is_numeric_argument,
    is_numeric_tuple,
    is_dolfin_exp,
)
from collections import OrderedDict
from firecrest.misc.time_storage import TimeSeries
from decimal import Decimal
import logging

DEFAULT_DT = 1.0e-3


class UnsteadyTVAcousticSolver(BaseSolver):
    def __init__(self, domain, **kwargs):
        super().__init__(domain)
        self.timer = kwargs.get("timer", None)
        if self.timer:
            self._dt = float(self.timer.get("dt", DEFAULT_DT))
        else:
            self._dt = kwargs.get("dt", DEFAULT_DT)

        self.forms = TVAcousticWeakForm(domain, **kwargs)

        self.LUSolver = None
        self.bilinear_form = None

        self._initial_state = None
        self.is_linearised = False

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
                    "Numeric argument / iterable, or dolf.Constant / dolf.Expression expected."
                    + f" Got {type(component)} instead."
                )
        self._initial_state = tuple(state_list)

    @property
    def _inverse_dt(self):
        return dolf.Constant(1.0 / self._dt)

    def _implicit_euler(self, initial_state):
        return self._theta_scheme(initial_state, theta=1.0)

    def _crank_nicolson(self, initial_state):
        return self._theta_scheme(initial_state, theta=0.5)

    def _theta_scheme(self, initial_state, theta):
        temporal_component = self.forms.temporal_component()
        temporal_component_old = self.forms.temporal_component(initial_state)

        spatial_component = self.forms.spatial_component()
        boundary_component = self.forms.boundary_components()
        dirichlet_bcs = self.forms.dirichlet_boundary_conditions(self.is_linearised)

        spatial_component_old = self.forms.spatial_component(initial_state)
        boundary_component_old = self.forms.boundary_components(initial_state)

        form = (
            self._inverse_dt * (temporal_component - temporal_component_old)
            + dolf.Constant(theta) * (spatial_component + boundary_component)
            + dolf.Constant(1 - theta)
            * (spatial_component_old + boundary_component_old)
        )

        return form, dirichlet_bcs

    def solve(self, initial_state, time_scheme="crank_nicolson"):
        try:
            solving_scheme = getattr(self, "_" + time_scheme)
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

    def solve_direct(
        self,
        initial_state,
        time_scheme="crank_nicolson",
        verbose=False,
        yield_state=False,
        plot_every=10,
    ):
        current_time = Decimal("0")
        final_time = self.timer["T"]
        state = TimeSeries(initial_state, current_time)
        self.LUSolver = None

        while current_time < final_time - Decimal(1.0e-8):
            w = self.solve(state.last, time_scheme=time_scheme)
            # This is a workaround to delete unused data. Otherwise
            # too much RAM is used
            try:
                state[current_time] = None
            except:
                pass
            current_time += self.timer["dt"]
            state[current_time] = w.split()

            if yield_state:
                yield state[current_time]

            if int(current_time / self.timer["dt"]) % plot_every == plot_every - 1:
                self.output_field(state[current_time])

            if verbose:
                print(
                    "Timestep: \t {0:.4f}->{1:.4f}".format(
                        current_time - self.timer["dt"], current_time
                    )
                )

        yield state

    def solve_adjoint(
        self,
        initial_state,
        time_scheme="crank_nicolson",
        verbose=False,
        yield_state=False,
        plot_every=10,
    ):
        """
        Solving the adjoint problem backwards in time. We reuse the direct forms, therefore
        the adjoint problem must be modified (see time symmetry in the unsteady control paper).
        The first time step is of length dt/2 for Crank-Nicolson, and brings us to the adjoint initial condition.

        :param initial_state: Direct state at the final time
        :param time_scheme: time discretization name
        :param verbose: verbosity level
        :return: adjoint time stepping history
        """
        if not time_scheme == "crank_nicolson":
            raise NotImplementedError(
                "Only crank_nicolson time scheme is implemented for direct-adjoint looping."
            )
        current_time = self.timer["T"]
        final_time = Decimal(self._dt)
        current_state = initial_state
        # I reset the factorization for the adjoint solver
        self.LUSolver = None
        self.is_linearised = True

        state = TimeSeries()

        # Half stepping first
        self._dt = self._dt / 2.0

        # form, bcs = self._implicit_euler(current_state)
        # linear_form = dolf.assemble(dolf.rhs(form))
        # bilinear_form = dolf.assemble(dolf.lhs(form))
        # for bc in bcs:
        #     bc.apply(bilinear_form)
        #     bc.apply(linear_form)
        # w = dolf.Function(self.forms.function_space)
        # dolf.solve(bilinear_form, w.vector(), linear_form)

        w = self.solve(current_state, "implicit_euler")
        self.LUSolver = None

        current_state = w.split(True)
        # if yield_state:
        #     yield current_state
        current_time -= self.timer["dt"] / Decimal("2")
        state[current_time] = current_state
        self._dt = self._dt * 2.0
        if yield_state:
            yield current_state

        # Regular time stepping
        while current_time > final_time + Decimal(1.0e-8):
            w = self.solve(current_state, time_scheme=time_scheme)

            current_state = w.split()
            if yield_state:
                yield current_state

            if int(float(current_time) / self._dt) % plot_every == plot_every - 1:
                self.output_field(current_state)

            if verbose:
                print(
                    "Timestep: \t {0:.4f}->{1:.4f}".format(
                        current_time, current_time - self.timer["dt"]
                    )
                )
            # This is a workaround to delete unused data. Otherwise
            # too much RAM is used
            try:
                state[current_time] = None
            except:
                pass
            current_time -= self.timer["dt"]
            state[current_time] = current_state

        self.is_linearised = False
        self.LUSolver = None

        yield state

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
