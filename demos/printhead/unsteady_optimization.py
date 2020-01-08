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
from firecrest.models.geometry_registry.symmetric_printhead_assembler import (
    SymmetricPrintheadGeometryAssembler,
)
from firecrest.models.free_surface_cap import Constants
from firecrest.misc.input_argparser import parser

decimal.getcontext().prec = 6

args = parser.parse_args()

setup_data = json.load(args.filename)

# Run mode
run_mode = setup_data["mode"]

# Logging settings
logging_data = setup_data["logging"]
plot_every = logging_data["plot_frequency"]
optimization_log_filename = logging_data["optimisation"]
energy_history_log_filename = logging_data["energy_history"]

# Geometry data for acoustic domain
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

# Time domain data

printhead_timescale = Decimal("0.1")  # in microseconds
time_domain_data = setup_data["time_domain"]
final_time = Decimal(str(time_domain_data["final_time"]))  # in microseconds
time_step = Decimal(str(time_domain_data["dt"]))  # in microseconds
nondim_final_time = final_time / printhead_timescale
nondim_time_step = time_step / printhead_timescale

timer = {"dt": nondim_time_step, "T": nondim_final_time}

# Waveform control data
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

geometry_assembler = SymmetricPrintheadGeometryAssembler(geometry_data)
domain = geometry_assembler.domain
control_boundary = geometry_assembler.control_boundary
shared_boundary = geometry_assembler.shared_boundary

experiment_id = (
    "act_" + str(geometry_data["actuator_length"]) + "_window_" + str(waveform_window)
)
print(experiment_id)


class NormalInflow:
    def __init__(self, series: TimeSeries):
        self.counter = 1
        self.series = list(series.values())

    def eval(self):
        if self.counter < len(self.series):
            value = (0.0, self.series[self.counter])
            self.counter += 1
            return value
        return 0.0, 0.0


class OptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_basis = PiecewiseLinearBasis(
            np.array([float(key) for key in default_grid.keys()]),
            width=kwargs.get("signal_window"),
            reduced_basis=True,
        )

    def flow_rate(self, state, boundary=shared_boundary):
        return 2.0 * dolf.assemble(
            dolf.inner(state[1], self.domain.n)
            * self.domain.ds((boundary.surface_index,))
        )

    def _objective_state(self, control):
        restored_control = self.linear_basis.extrapolate(list(control))

        control_boundary.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list(restored_control, default_grid)
        )

        surface_model = SurfaceModel(nondim_constants, kappa_t0=initial_curvature)
        shared_boundary.bcond["normal_force"] = surface_model

        _old_flow_rate = 0
        for state in self.solve_direct(
            initial_state, verbose=False, yield_state=True, plot_every=plot_every
        ):

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
                                state, (0.0, -1.0), shared_boundary
                            ),
                            self.flow_rate(state, control_boundary),
                            self.flow_rate(state, shared_boundary),
                        ]
                    )

            _new_flow_rate = self.flow_rate(state)
            surface_model.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )
            _old_flow_rate = _new_flow_rate

        if optimization_log_filename and run_mode == "optimization":
            file_name = optimization_log_filename + experiment_id + ".dat"
            with open(file_name, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self._objective((state, surface_model), verbose=False),
                        surface_model.kappa,
                    ]
                    + list(control)
                )

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

    def boundary_averaged_stress(self, adjoint_state):
        stress = self.forms.stress(adjoint_state[0], adjoint_state[1])
        stress = dolf.assemble(
            dolf.dot(dolf.dot(stress, self.domain.n), self.domain.n)
            * self.domain.ds((control_boundary.surface_index,))
        )
        return stress

    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        shared_boundary.bcond["normal_force"] = adjoint_surface

        adjoint_stress_averaged = midpoint_grid
        current_time = self.timer["T"] - self.timer["dt"] / Decimal("2")

        _old_flow_rate = 0
        _new_flow_rate = 0
        adjoint_surface.update_curvature(
            0.5 * (_new_flow_rate + _old_flow_rate), self._dt
        )
        for adjoint_state in solver.solve_adjoint(
            initial_state=state, verbose=False, yield_state=True, plot_every=plot_every
        ):
            _old_flow_rate = _new_flow_rate
            _new_flow_rate = self.flow_rate(adjoint_state)
            adjoint_surface.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )

            adjoint_stress_averaged[current_time] = self.boundary_averaged_stress(
                adjoint_state
            )
            current_time -= self.timer["dt"]

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0.0
        du[timer["T"]] = 0.0

        discrete_grad = self.linear_basis.discretize(du.values())

        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())

        print(
            "discrete gradient norm: ",
            (
                adjoint_stress_averaged
                * TimeSeries.from_list(
                    self.linear_basis.extrapolate(discrete_grad), default_grid
                )
            ).integrate(),
        )

        return discrete_grad


default_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)
small_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(1, int(timer["T"] / Decimal(timer["dt"])))
    }
)
midpoint_grid = TimeSeries.from_dict(
    {
        Decimal(k + 0.5) * Decimal(timer["dt"]): 0
        for k in range(int(timer["T"] / Decimal(timer["dt"])) - 1)
    }
)

solver = OptimizationSolver(
    domain,
    Re=Re,
    Pr=Pr,
    timer=timer,
    signal_window=nondim_waveform_window,
    experiment_id=experiment_id,
    optimization_method=control_algorithm,
)
initial_state = (0.0, (0.0, 0.0), 0.0)


x0 = [0.0 for _ in range(len(default_grid))]
top_bound = [control_upper_limit for i in range(len(x0))]
low_bound = [control_lower_limit for i in range(len(x0))]
top_bound = solver.linear_basis.discretize(top_bound)
low_bound = solver.linear_basis.discretize(low_bound)
bnds = list(zip(low_bound, top_bound))
if control_default_value:
    x0 = control_default_value
else:
    x0 = solver.linear_basis.discretize(x0)

if run_mode == "taylor_test":
    energy = []
    _state = solver._objective_state(x0)
    energy.append(solver._objective(_state))
    print(energy[-1])
    grad = solver._jacobian(_state)

    for i in range(1, 11):
        new_state = solver._objective_state(grad * 1.0e-3 * i)
        energy.append(solver._objective(new_state))
elif run_mode == "optimization":
    res = solver.minimize(x0, bnds)
elif run_mode == "single_run":
    solver._objective_state(x0)
else:
    raise NotImplementedError(f"No mode called {run_mode} is implemented")
