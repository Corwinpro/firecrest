from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel, AdjointSurfaceModel
import dolfin as dolf
from collections import namedtuple
from decimal import Decimal
from firecrest.misc.time_storage import TimeSeries
from firecrest.misc.optimization_mixin import OptimizationMixin
import numpy as np

from firecrest.mesh.unit_domain import IntervalDomain, PointBoundary


p1 = PointBoundary(
    [0], inside=lambda x: x[0] < 0.1, bcond={"normal_force": None, "adiabatic": True}
)
p2 = PointBoundary(
    [1], inside=lambda x: x[0] > 0.9, bcond={"inflow": 0.0, "adiabatic": True}
)
domain = IntervalDomain([p1, p2], resolution=1000)

elsize = 0.08
height = 0.7
length = 9.2
actuator_length = 4.0
offset_top = 1.0
nozzle_r = 0.1
nozzle_l = 0.2
nozzle_offset = 0.2
manifold_width = 2.0
manifold_height = 4.7

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
printhead_constants = Constants(rho, epsilon, L, 10.0e-6, 10.0e-6, gamma_st, Re, c_s)

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


class NormalInflow:
    def __init__(self, series: TimeSeries):
        self.counter = 1
        self.series = list(series.values())

    def eval(self):
        if self.counter < len(self.series):
            value = self.series[self.counter]
            self.counter += 1
            return value
        return 0.0


class OptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def flow_rate(self, state):
        return dolf.assemble(
            dolf.inner(state[1], self.domain.n) * self.domain.ds((p1.surface_index,))
        )

    def _objective_state(self, control):
        p2.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list([0.0] + list(control) + [0.0], default_grid)
        )

        surface_model = SurfaceModel(nondim_constants, kappa_t0=0.05)
        p1.bcond["normal_force"] = surface_model

        _old_flow_rate = 0
        for state in self.solve_direct(
            initial_state, verbose=False, yield_state=True, plot_every=1000
        ):
            if isinstance(state, TimeSeries):
                direct_history = state
                break

            # with open("energy.dat", "a") as file:
            #     _str = (
            #         str(self.forms.energy(state))
            #         + " "
            #         + str(surface_model.surface_energy())
            #     )
            #     file.write(_str)
            #     file.write("\n")

            _new_flow_rate = self.flow_rate(state)
            surface_model.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )
            _old_flow_rate = _new_flow_rate

        with open("log.dat", "a") as file:
            file.write(
                str(
                    self._objective((direct_history.last, surface_model), verbose=False)
                )
                + "  "
                + str(surface_model.kappa)
                + ":\t"
            )
            for c in control:
                file.write(str(c) + ",")
            file.write("\n")

        return direct_history.last, surface_model

    def _objective(self, state, verbose=True):
        if verbose:
            print(
                "evaluated at: ",
                self.forms.energy(state[0]) + state[1].surface_energy(state[1].kappa),
                " curvature: ",
                state[1].kappa,
            )
        return self.forms.energy(state[0]) + state[1].surface_energy(state[1].kappa)

    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        p1.bcond["normal_force"] = adjoint_surface

        _old_flow_rate = 0
        _new_flow_rate = 0
        adjoint_surface.update_curvature(
            0.5 * (_new_flow_rate + _old_flow_rate), self._dt
        )
        for adjoint_state in solver.solve_adjoint(
            initial_state=state, verbose=False, yield_state=True
        ):
            if isinstance(adjoint_state, TimeSeries):
                adjoint_history = adjoint_state
                break
            _old_flow_rate = _new_flow_rate
            _new_flow_rate = self.flow_rate(adjoint_state)
            adjoint_surface.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )

        adjoint_stress = adjoint_history.apply(lambda x: self.forms.stress(x[0], x[1]))
        adjoint_stress_averaged = adjoint_stress.apply(
            lambda x: dolf.assemble(
                dolf.dot(dolf.dot(x, self.domain.n), self.domain.n)
                * self.domain.ds((p2.surface_index,))
            )
        )

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0.0
        du[timer["T"]] = 0.0

        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())

        return np.array(du.values()[1:-1]) * self._dt ** 0.5


timer = {"dt": Decimal("0.01"), "T": Decimal("3.0")}
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

surface_model = SurfaceModel(nondim_constants, kappa_t0=0.05)
solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)
initial_state = (0.0, 0.0, 0.0)


x0 = [0.0 for _ in range(len(default_grid) - 2)]
top_bound = [0.03 for i in range(len(x0))]
low_bound = [-0.01 for i in range(len(x0))]
bnds = list(zip(low_bound, top_bound))


run_taylor_test = False
if run_taylor_test:
    energy = []
    state = solver._objective_state(x0)
    energy.append(solver._objective(state))
    print(energy[-1])
    grad = np.array(solver._jacobian(state))

    for i in range(1, 11):
        new_state = solver._objective_state(grad * 1.0e-5 * i)
        energy.append(solver._objective(new_state))

    exit()

res = solver.minimize(x0, bnds)