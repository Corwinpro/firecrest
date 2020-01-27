import dolfin as dolf
from decimal import Decimal
import numpy as np
import logging

from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.misc.time_storage import TimeSeries, PiecewiseLinearBasis
from firecrest.misc.optimization_mixin import OptimizationMixin

log = logging.getLogger("UnsteadyOptimizationSolver")


class NormalInflow:
    def __init__(self, control_data):
        self.counter = 1
        self.control_data = control_data

    def eval(self):
        if self.counter < len(self.control_data):
            value = (0.0, self.control_data[self.counter])
            self.counter += 1
            return value
        return 0.0, 0.0


def discrete_to_continuous_direct(function):
    def wrapped(self, discrete_control):
        continuous_control = self.basis.extrapolate(list(discrete_control))
        return function(self, continuous_control)

    return wrapped


def post(function):
    def wrapped(self, discrete_control):
        result = function(self, discrete_control)

        if self.logger.if_log_data("optimization"):
            output_data = [
                self._objective(result, verbose=False),
                result[1].kappa,
            ] + list(discrete_control)
            self.logger.log_optimization_step(output_data)
        # if optimization_log_filename and run_mode == "optimization":
        #     file_name = optimization_log_filename + experiment_id + ".dat"
        #     output_data = [
        #         self._objective(result, verbose=False),
        #         result[1].kappa,
        #     ] + list(discrete_control)
        #     with open(file_name, "a") as file:
        #         writer = csv.writer(file)
        #         writer.writerow(output_data)

        return result

    return wrapped


def continuous_to_discrete_adj(function):
    def wrapped(self, state):
        continuous_gradient = function(self, state)
        gradient_time_series = TimeSeries.from_list(
            continuous_gradient, self.midpoint_grid
        )
        du = TimeSeries.interpolate_to_keys(gradient_time_series, self.time_grid)

        discrete_grad = self.basis.discretize(du.values())

        log.info(f"gradient norm: {(gradient_time_series * du).integrate()}")
        log.info(
            "discrete gradient norm: {}".format(
                (
                    gradient_time_series
                    * TimeSeries.from_list(
                        self.basis.extrapolate(discrete_grad), self.time_grid
                    )
                ).integrate()
            )
        )
        return discrete_grad

    return wrapped


class UnsteadyOptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state = kwargs["initial_state"]
        self.control_boundary = kwargs.get("control_boundary")
        self.shared_boundary = kwargs.get("shared_boundary")

        self.logger = kwargs.get("logger")
        self.model_factory = kwargs.get("model_factory")

    def flow_rate(self, state, boundary):
        return 2.0 * dolf.assemble(
            dolf.inner(state[1], self.domain.n)
            * self.domain.ds((boundary.surface_index,))
        )

    def initialize_control_boundary(self, control_data):
        self.control_boundary.bcond["inflow"] = NormalInflow(control_data)

    def log_intermediate_step(self, state, surface_model):
        if self.logger.if_log_data("in_process"):
            data = [
                self.forms.energy(state),
                surface_model.surface_energy() / 2.0,
                self.forms.kinetic_energy_flux(
                    state, (0.0, 1.0), self.control_boundary
                ),
                self.forms.kinetic_energy_flux(
                    state, (0.0, -1.0), self.shared_boundary
                ),
                self.flow_rate(state, self.control_boundary),
                self.flow_rate(state, self.shared_boundary),
            ]
            self.logger.log_intermediate_step(data)

        # if energy_history_log_filename and run_mode == "single_run":
        #     file_name = energy_history_log_filename + experiment_id + ".dat"
        #     with open(file_name, "a") as file:
        #         writer = csv.writer(file)
        #         writer.writerow(
        #             [
        #                 self.forms.energy(state),
        #                 surface_model.surface_energy() / 2.0,
        #                 self.forms.kinetic_energy_flux(
        #                     state, (0.0, 1.0), self.control_boundary
        #                 ),
        #                 self.forms.kinetic_energy_flux(
        #                     state, (0.0, -1.0), self.shared_boundary
        #                 ),
        #                 self.flow_rate(state, self.control_boundary),
        #                 self.flow_rate(state, self.shared_boundary),
        #             ]
        #         )

    def initialize_shared_boundary(self):
        # surface_model = SurfaceModel(nondim_constants, kappa_t0=initial_curvature)
        surface_model = self.model_factory.create_direct_model()
        if "normal_force" in self.shared_boundary.bcond:
            self.shared_boundary.bcond["normal_force"] = surface_model
        return surface_model

    def initialize_adjoint_shared_boundary(self, direct_shared_boundary):
        # adjoint_surface = AdjointSurfaceModel(direct_surface=direct_shared_boundary)
        adjoint_surface = self.model_factory.create_adjoint_model(
            direct_shared_boundary
        )
        self.shared_boundary.bcond["normal_force"] = adjoint_surface
        adjoint_surface.update_curvature(0.0, self._dt)
        return adjoint_surface

    def evaluate_adjoint_sensitivity(self, adjoint_state):
        return self.boundary_averaged_stress(adjoint_state, self.control_boundary)

    @post
    @discrete_to_continuous_direct
    def _objective_state(self, control):
        self.initialize_control_boundary(control)

        surface_model = self.initialize_shared_boundary()
        print(self.initial_state)
        _old_flow_rate = 0
        for state in self.solve_direct(
            self.initial_state,
            verbose=False,
            yield_state=True,
            plot_every=self.logger.plot_every,
        ):
            _new_flow_rate = self.flow_rate(state, self.shared_boundary)
            surface_model.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )
            _old_flow_rate = _new_flow_rate

            self.log_intermediate_step(state, surface_model)

        return state, surface_model

    def _objective(self, state, verbose=True):
        acoustic_energy = self.forms.energy(state[0])
        free_energy = state[1].surface_energy(state[1].kappa) / 2.0
        if verbose:
            log.info(
                f"evaluated at: {acoustic_energy + free_energy}"
                f" curvature: {state[1].kappa}"
            )
        return acoustic_energy + free_energy

    def boundary_averaged_stress(self, state, boundary):
        stress = self.forms.stress(state[0], state[1])
        stress = dolf.assemble(
            dolf.dot(dolf.dot(stress, self.domain.n), self.domain.n)
            * self.domain.ds((boundary.surface_index,))
        )
        return stress

    @continuous_to_discrete_adj
    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = self.initialize_adjoint_shared_boundary(surface_model)

        continuous_gradient = []

        _old_flow_rate = 0
        _new_flow_rate = 0
        for adjoint_state in self.solve_adjoint(
            initial_state=state,
            verbose=False,
            yield_state=True,
            plot_every=self.logger.plot_every,
        ):
            _old_flow_rate = _new_flow_rate
            _new_flow_rate = self.flow_rate(adjoint_state, self.shared_boundary)
            adjoint_surface.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )
            continuous_gradient.append(self.evaluate_adjoint_sensitivity(adjoint_state))

        return continuous_gradient[::-1]

    def run(
        self,
        initial_guess=None,
        bounds=None,
        signal_window=None,
        optimization_method=None,
    ):
        # 1. define control
        # `time_grid` defines the time grid for the direct simulation
        self.time_grid = TimeSeries.from_dict(
            {
                Decimal(k) * Decimal(self.timer["dt"]): 0
                for k in range(int(self.timer["T"] / Decimal(self.timer["dt"])) + 1)
            }
        )
        # `midpoint_grid` defines the time grid for the adjoint simulation
        self.midpoint_grid = TimeSeries.from_dict(
            {
                Decimal(k + 0.5) * Decimal(self.timer["dt"]): 0
                for k in range(int(self.timer["T"] / Decimal(self.timer["dt"])))
            }
        )
        # We define a control space, a PiecewiseLinearBasis in this case
        self.basis = PiecewiseLinearBasis(
            np.array([float(key) for key in self.time_grid.keys()]),
            width=signal_window,
            reduced_basis=True,
        )

        # We prepare the initial guess and bounds on the control
        if bounds is not None:
            low_bound = [bounds[0] for _ in range(len(self.time_grid))]
            top_bound = [bounds[1] for _ in range(len(self.time_grid))]
            top_bound = self.basis.discretize(top_bound)
            low_bound = self.basis.discretize(low_bound)
            bounds = list(zip(low_bound, top_bound))

        if initial_guess is None:
            initial_guess = np.zeros(len(self.basis.basis))

        if self.logger.run_mode == "optimization":
            res = self.minimize(
                initial_guess, bounds, optimization_method=optimization_method
            )
        elif self.logger.run_mode == "single_run":
            res = self._objective_state(initial_guess)
        return res
