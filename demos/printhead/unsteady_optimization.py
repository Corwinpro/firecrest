import dolfin as dolf
import decimal
from decimal import Decimal
import numpy as np
import json
import csv

from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel, AdjointSurfaceModel
from firecrest.misc.time_storage import TimeSeries, PiecewiseLinearBasis
from firecrest.misc.optimization_mixin import OptimizationMixin
from firecrest.models.geometry_registry.geometry_registry import geometry_registry
from firecrest.models.free_surface_cap import Constants
from firecrest.misc.input_argparser import parser

decimal.getcontext().prec = 6

args = parser.parse_args()

setup_data = json.load(args.filename)

# Run mode +
run_mode = setup_data["mode"]

# Logging settings +
logging_data = setup_data["logging"]
plot_every = logging_data["plot_frequency"]
optimization_log_filename = logging_data["optimisation"]
energy_history_log_filename = logging_data["energy_history"]

# Geometry data for acoustic domain +
geometry_data = setup_data["acoustic_domain"]

# Dimensional constants parse
constants_data = setup_data["constants"]
L = constants_data["length"]
c_s = constants_data["sound_speed"]
rho = constants_data["density"]
epsilon = constants_data["Mach"]
gamma_st = constants_data["surface_tension"]
mu = constants_data["viscosity"]
Pr = constants_data["Pr"]
Re = rho * c_s * L / mu

# Nozzle domain constants
nozzle_domain_data = setup_data["nozzle_domain"]
initial_curvature = nozzle_domain_data["initial_curvature"]
nozzle_domain_length = nozzle_domain_data["length"]
nozzle_domain_radius = nozzle_domain_data["radius"]

# Time domain data +
printhead_timescale = Decimal("0.1")  # in microseconds
time_domain_data = setup_data["time_domain"]
final_time = Decimal(str(time_domain_data["final_time"]))  # in microseconds
time_step = Decimal(str(time_domain_data["dt"]))  # in microseconds
nondim_final_time = final_time / printhead_timescale
nondim_time_step = time_step / printhead_timescale

timer = {"dt": nondim_time_step, "T": nondim_final_time}

# Waveform control data +
waveform_data = setup_data["control_space"]
control_type = waveform_data["type"]
waveform_window = waveform_data["window"]  # in microseconds
nondim_waveform_window = waveform_window / float(printhead_timescale)
control_default_value = waveform_data.get("control_default", None)
control_upper_limit = waveform_data.get("upper_limit", 0.015)
control_lower_limit = waveform_data.get("lower_limit", -0.015)
control_algorithm = waveform_data.get("algorithm", "L-BFGS-B")


nondim_constants = Constants(
    rho / rho,
    epsilon,
    L / L,
    nozzle_domain_length / L,
    nozzle_domain_radius / L,
    2.0 * gamma_st / (rho * c_s ** 2.0 * nozzle_domain_radius * epsilon),
    Re,
    c_s / c_s,
)

# +
geometry_type = geometry_data.get("type", "symmetric_printhead")
geometry_assembler = geometry_registry[geometry_type](geometry_data)
domain = geometry_assembler.domain
control_boundary = geometry_assembler.control_boundary
shared_boundary = geometry_assembler.shared_boundary

# +
experiment_id = (
    geometry_type
    + "_act_"
    + str(geometry_data["actuator_length"])
    + "_window_"
    + str(waveform_window)
    + "_T_"
    + str(final_time)
)
print(experiment_id)


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


def discrete_to_continuous_control(function):
    def wrapped(self, discrete_control):
        continuous_control = self.linear_basis.extrapolate(list(discrete_control))
        return function(self, continuous_control)

    return wrapped


def post(function):
    def wrapped(self, discrete_control):
        result = function(self, discrete_control)

        if optimization_log_filename and run_mode == "optimization":
            file_name = optimization_log_filename + experiment_id + ".dat"
            output_data = [
                self._objective(result, verbose=False),
                result[1].kappa,
            ] + list(discrete_control)
            with open(file_name, "a") as file:
                writer = csv.writer(file)
                writer.writerow(output_data)

        return result

    return wrapped


def continuous_to_discrete_adj(function):
    def wrapped(self, state):
        continuous_gradient = function(self, state)
        gradient_time_series = TimeSeries.from_list(
            continuous_gradient, self.midpoint_grid
        )
        du = TimeSeries.interpolate_to_keys(gradient_time_series, self.time_grid)

        discrete_grad = self.linear_basis.discretize(du.values())

        print("gradient norm: ", (gradient_time_series * du).integrate())

        print(
            "discrete gradient norm: ",
            (
                gradient_time_series
                * TimeSeries.from_list(
                    self.linear_basis.extrapolate(discrete_grad), self.time_grid
                )
            ).integrate(),
        )
        return discrete_grad

    return wrapped


class OptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, signal_window=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_window = signal_window
        self.initial_state = kwargs["initial_state"]
        self.control_boundary = kwargs.get("control_boundary")
        self.shared_boundary = kwargs.get("shared_boundary")

    def flow_rate(self, state, boundary):
        return 2.0 * dolf.assemble(
            dolf.inner(state[1], self.domain.n)
            * self.domain.ds((boundary.surface_index,))
        )

    def initialize_control_boundary(self, control_data):
        inflow = NormalInflow(control_data)
        self.control_boundary.bcond["inflow"] = inflow

    def log_intermediate_step(self, state, surface_model):
        if energy_history_log_filename and run_mode == "single_run":
            file_name = energy_history_log_filename + experiment_id + ".dat"
            with open(file_name, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.forms.energy(state),
                        surface_model.surface_energy() / 2.0,
                        self.forms.kinetic_energy_flux(
                            state, (0.0, 1.0), control_boundary
                        ),
                        self.forms.kinetic_energy_flux(
                            state, (0.0, -1.0), self.shared_boundary
                        ),
                        self.flow_rate(state, self.control_boundary),
                        self.flow_rate(state, self.shared_boundary),
                    ]
                )

    def initialize_shared_boundary(self):
        surface_model = SurfaceModel(nondim_constants, kappa_t0=initial_curvature)
        if "normal_force" in self.shared_boundary.bcond:
            self.shared_boundary.bcond["normal_force"] = surface_model
        return surface_model

    @post
    @discrete_to_continuous_control
    def _objective_state(self, control):
        self.initialize_control_boundary(control)

        surface_model = self.initialize_shared_boundary()

        _old_flow_rate = 0
        for state in self.solve_direct(
            self.initial_state, verbose=False, yield_state=True, plot_every=plot_every
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
            print(
                "evaluated at: ",
                acoustic_energy + free_energy,
                " curvature: ",
                state[1].kappa,
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

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        self.shared_boundary.bcond["normal_force"] = adjoint_surface

        adjoint_stress_averaged = []

        _old_flow_rate = 0
        _new_flow_rate = 0
        adjoint_surface.update_curvature(
            0.5 * (_new_flow_rate + _old_flow_rate), self._dt
        )
        for adjoint_state in solver.solve_adjoint(
            initial_state=state, verbose=False, yield_state=True, plot_every=plot_every
        ):
            _old_flow_rate = _new_flow_rate
            _new_flow_rate = self.flow_rate(adjoint_state, self.shared_boundary)
            adjoint_surface.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )

            adjoint_stress_averaged.append(
                self.boundary_averaged_stress(adjoint_state, self.control_boundary)
            )

        return adjoint_stress_averaged[::-1]

    def run(self, initial_guess=None, bounds=None):
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
                Decimal(k + 0.5) * Decimal(timer["dt"]): 0
                for k in range(int(timer["T"] / Decimal(timer["dt"])) - 1)
            }
        )
        # We define a control space, a PiecewiseLinearBasis in this case
        self.linear_basis = PiecewiseLinearBasis(
            np.array([float(key) for key in self.time_grid.keys()]),
            width=self.signal_window,
            reduced_basis=True,
        )

        # We prepare the initial guess and bounds on the control
        if bounds is not None:
            top_bound = [bounds[0] for _ in range(len(self.time_grid))]
            low_bound = [bounds[1] for _ in range(len(self.time_grid))]
            top_bound = self.linear_basis.discretize(top_bound)
            low_bound = self.linear_basis.discretize(low_bound)
            bounds = list(zip(low_bound, top_bound))

        if initial_guess is None:
            initial_guess = np.zeros(len(self.linear_basis.basis))

        res = self.minimize(initial_guess, bounds)
        return res


solver = OptimizationSolver(
    domain,
    Re=Re,
    Pr=Pr,
    timer=timer,
    signal_window=nondim_waveform_window,
    experiment_id=experiment_id,
    optimization_method=control_algorithm,
    initial_state=(0.0, (0.0, 0.0), 0.0),
    control_boundary=geometry_assembler.control_boundary,
    shared_boundary=geometry_assembler.shared_boundary,
)
res = solver.run(control_default_value, (control_lower_limit, control_upper_limit))
#
# if run_mode == "taylor_test":
#     x0 = [1.0e-4 for _ in range(len(x0))]
#     energy = []
#     _state = solver._objective_state(x0)
#     energy.append(solver._objective(_state))
#     print(energy[-1])
#     grad = np.array(solver._jacobian(_state))
#     x0 = np.array(x0)
#
#     for i in range(1, 11):
#         new_state = solver._objective_state(x0 + grad * 1.0e-6 * i)
#         energy.append(solver._objective(new_state))
# elif run_mode == "optimization":
#     # res = solver.minimize(x0, bnds)
#     res = solver.run(control_default_value, (control_lower_limit, control_upper_limit))
# elif run_mode == "single_run":
#     solver._objective_state(x0)
# else:
#     raise NotImplementedError(f"No mode called {run_mode} is implemented")
