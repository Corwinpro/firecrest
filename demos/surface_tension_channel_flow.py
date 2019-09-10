from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel, AdjointSurfaceModel
import dolfin as dolf
from collections import namedtuple
from decimal import Decimal
from firecrest.misc.time_storage import TimeSeries, PiecewiseLinearBasis
from firecrest.misc.optimization_mixin import OptimizationMixin
import numpy as np

elsize = 0.05  # 04
height = 0.7
length = 5.0  # 10.0
offset_top = 1.0
nozzle_r = 0.1
nozzle_l = 0.2
nozzle_offset = 0.2

# Define constants
L = 1.0e-4
c_s = 1.0e3
rho = 1.0e3
epsilon = 1.0e-3
gamma_st = 50.0e-3
mu = 1.0e-2
Re = rho * c_s * L / mu
a1 = 4.0 / 3.0
a2 = 8
kappa_prime = epsilon / nozzle_r ** 4 * 0.25  # up to a constant

# non dimensional
alpha_1 = nozzle_l / (3.1415 * nozzle_r ** 2) * a1
alpha_2 = nozzle_l / (3.1415 * nozzle_r ** 3) / Re * a2 / nozzle_r
gamma_nd = 2 * gamma_st / (rho * c_s ** 2 * nozzle_r * epsilon)


Constants = namedtuple(
    "Constants",
    [
        "density",
        "acoustic_mach",
        "channel_L",
        "nozzle_L",
        "nozzle_R",
        "surface_tension",
        "Re",
        "sound_speed",
    ],
)
printhead_constants = Constants(
    1.0e3, 1.0e-3, 100.0e-6, 10.0e-6, 10.0e-6, 50.0e-3, 5.0e3, 1.0e3
)

nondim_constants = Constants(
    1.0,
    printhead_constants.acoustic_mach,
    1,
    printhead_constants.nozzle_L / printhead_constants.channel_L,
    printhead_constants.nozzle_R / printhead_constants.channel_L,
    2.0
    * printhead_constants.surface_tension
    / (
        printhead_constants.density
        * printhead_constants.sound_speed ** 2.0
        * printhead_constants.nozzle_R
        * printhead_constants.acoustic_mach
    ),
    printhead_constants.Re,
    1.0,
)

control_points_1 = [[offset_top, height], [1.0e-16, height]]
control_points_free_left = [[1.0e-16, height], [0.0, 0.0 + 1.0e-16]]
control_points_bot_left = [
    [0.0, 0.0 + 1.0e-16],
    [length / 2.0 - nozzle_r - nozzle_offset, 0.0],
]
control_points_refine_left = [
    [length / 2.0 - nozzle_r - nozzle_offset, 0.0],
    [length / 2.0 - nozzle_r, 0.0],
    [length / 2.0 - nozzle_r, -nozzle_l],
]
control_points_2 = [
    [length / 2.0 - nozzle_r, -nozzle_l],
    [length / 2.0 + nozzle_r, -nozzle_l + 1.0e-16],
]
control_points_refine_right = [
    [length / 2.0 + nozzle_r, -nozzle_l + 1.0e-16],
    [length / 2.0 + nozzle_r, 0.0],
    [length / 2.0 + nozzle_r + nozzle_offset, 0.0],
]
control_points_bot_right = [
    [length / 2.0 + nozzle_r + nozzle_offset, 0.0],
    [length, 0.0 + 1.0e-16],
]
control_points_free_right = [[length, 0.0 + 1.0e-16], [length, height]]
control_points_3 = [[length, height], [length - offset_top, height + 1.0e-16]]
control_points_4 = [[length - offset_top, height + 1.0e-16], [offset_top, height]]


boundary1 = LineElement(
    control_points_1, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_free_left = LineElement(
    control_points_free_left, el_size=elsize, bcond={"free": True, "adiabatic": True}
)
boundary_bot_left = LineElement(
    control_points_bot_left, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_refine_left = LineElement(
    control_points_refine_left,
    el_size=elsize / 4.0,
    bcond={"noslip": True, "adiabatic": True},
)
boundary2 = LineElement(
    control_points_2,
    el_size=elsize / 4.0,
    bcond={"normal_force": None, "adiabatic": True},
)
boundary_refine_right = LineElement(
    control_points_refine_right,
    el_size=elsize / 4.0,
    bcond={"noslip": True, "adiabatic": True},
)
boundary_bot_right = LineElement(
    control_points_bot_right, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_free_right = LineElement(
    control_points_free_right, el_size=elsize, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4, el_size=elsize, bcond={"inflow": (0.0, 0.0), "adiabatic": True}
)
domain_boundaries = (
    boundary1,
    boundary_free_left,
    boundary_bot_left,
    boundary_refine_left,
    boundary2,
    boundary_refine_right,
    boundary_bot_right,
    boundary_free_right,
    boundary3,
    boundary4,
)
domain = SimpleDomain(domain_boundaries)


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
            width=kwargs.get("signal_window", 0.25),
        )

    def _objective_state(self, control):
        restored_control = self.linear_basis.extrapolate([0.0] + list(control) + [0.0])
        boundary4.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list(restored_control, default_grid)
        )

        surface_model = SurfaceModel(nondim_constants, kappa_t0=0.25)
        boundary2.bcond["normal_force"] = surface_model

        for state in self.solve_direct(initial_state, verbose=False, yield_state=True):
            if isinstance(state, TimeSeries):
                direct_history = state
                break
            flow_rate = dolf.assemble(
                dolf.inner(state[1], domain.n) * domain.ds((boundary2.surface_index,))
            )
            surface_model.update_curvature(flow_rate, self._dt)

        with open("log.dat", "a") as file:
            file.write(
                str(self._objective((direct_history.last, surface_model))) + ":\t"
            )
            for c in control:
                file.write(str(c) + " ")
            file.write("\n")

        return direct_history.last, surface_model

    def _objective(self, state):
        print(
            "evaluated at: ",
            self.forms.energy(state[0]) + state[1].surface_energy(),
            " curvature: ",
            state[1].kappa,
        )
        return self.forms.energy(state[0]) + state[1].surface_energy()

    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        boundary2.bcond["normal_force"] = adjoint_surface

        for adjoint_state in solver.solve_adjoint(
            initial_state=state, verbose=False, yield_state=True
        ):
            if isinstance(adjoint_state, TimeSeries):
                adjoint_history = adjoint_state
                break
            flow_rate = dolf.assemble(
                dolf.inner(adjoint_state[1], domain.n)
                * domain.ds((boundary2.surface_index,))
            )
            adjoint_surface.update_curvature(flow_rate, self._dt)

        adjoint_stress = adjoint_history.apply(lambda x: self.forms.stress(x[0], x[1]))
        adjoint_stress_averaged = adjoint_stress.apply(
            lambda x: dolf.assemble(
                dolf.dot(dolf.dot(x, self.domain.n), self.domain.n)
                * self.domain.ds((boundary4.surface_index,))
            )
        )

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0
        du[timer["T"]] = 0

        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())

        discrete_grad = self.linear_basis.discretize(du.values())
        default_grid[0] = 0.0
        discrete_grad[-1] = 0.0
        print(
            "discrete gradient norm: ",
            self.linear_basis.mass_matrix.dot(np.array(discrete_grad)).dot(
                discrete_grad
            )
            * self._dt,
        )

        return discrete_grad[1:-1]


timer = {"dt": Decimal("0.01"), "T": Decimal("20.0")}
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

surface_model = SurfaceModel(nondim_constants, kappa_t0=0.25)


solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)
initial_state = (0.0, (0.0, 0.0), 0.0)
x0 = [0.0 for _ in range(len(default_grid))]
x0 = solver.linear_basis.discretize(x0)[1:-1]
bnds = tuple((-0.05, 0.05) for i in range(len(x0)))
res = solver.minimize(x0, bnds)

# solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)
# x0 = [0.0 for _ in range(len(default_grid) - 2)]
# direct_last = solver._objective_state(x0)
# print(solver._objective(direct_last))
# control = solver._jacobian(direct_last)
# print(control)
#
# direct_last = solver._objective_state(control)
# print(solver._objective(direct_last))
# for state in solver.solve_direct(initial_state, verbose=True, yield_state=True):
#     if isinstance(state, TimeSeries):
#         direct_history = state
#         break
#     flow_rate = dolf.assemble(
#         dolf.inner(state[1], domain.n) * domain.ds((boundary2.surface_index,))
#     )
#     surface_model.update_curvature(flow_rate, solver._dt)
#
#
# last = direct_history.last
# last = (last[0], -last[1], last[2])
# adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
# boundary2.bcond["normal_force"] = adjoint_surface
#
# for adjoint_state in solver.solve_adjoint(
#     initial_state=last, verbose=True, yield_state=True
# ):
#     if isinstance(adjoint_state, TimeSeries):
#         adjoint_history = adjoint_state
#         break
#     flow_rate = dolf.assemble(
#         dolf.inner(adjoint_state[1], domain.n) * domain.ds((boundary2.surface_index,))
#     )
#     adjoint_surface.update_curvature(flow_rate, solver._dt)
