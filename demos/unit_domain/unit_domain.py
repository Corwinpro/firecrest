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
domain = IntervalDomain([p1, p2], resolution=100)
timer = {"dt": Decimal("0.001"), "T": Decimal("0.1")}


def resolve_mesh(mesh, iterations, cut):
    for _ in range(iterations):
        markers = dolf.MeshFunction("bool", mesh, False)
        for c in dolf.cells(mesh):
            if c.midpoint().x() > cut:
                markers[c] = True

        mesh = dolf.refine(mesh, markers)
    return mesh


# domain.mesh = resolve_mesh(domain.mesh, 1)

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

grad_data = {"grad_norm": None}


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
            initial_state, verbose=False, yield_state=True, plot_every=100000
        ):
            # with open("energy_25.dat", "a") as file:
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

        # with open("log.dat", "a") as file:
        #     file.write(
        #         str(self._objective((state, surface_model), verbose=False))
        #         + "  "
        #         + str(surface_model.kappa)
        #         + ":\t"
        #     )
        #     for c in control:
        #         file.write(str(c) + ",")
        #     file.write("\n")

        return state, surface_model

    def _objective(self, state, verbose=True):
        if verbose:
            print(
                "evaluated at: ",
                self.forms.energy(state[0]) + state[1].surface_energy(state[1].kappa),
                " curvature: ",
                state[1].kappa,
            )
        return self.forms.energy(state[0]) + state[1].surface_energy(state[1].kappa)

    def boundary_averaged_stress(self, adjoint_state):
        stress = self.forms.stress(adjoint_state[0], adjoint_state[1])
        stress = dolf.assemble(
            dolf.dot(dolf.dot(stress, self.domain.n), self.domain.n)
            * self.domain.ds((p2.surface_index,))
        )
        return stress

    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        p1.bcond["normal_force"] = adjoint_surface
        adjoint_stress_averaged = midpoint_grid
        current_time = self.timer["T"] - self.timer["dt"] / Decimal("2")

        _old_flow_rate = 0
        _new_flow_rate = 0
        adjoint_surface.update_curvature(
            0.5 * (_new_flow_rate + _old_flow_rate), self._dt
        )
        for adjoint_state in solver.solve_adjoint(
            initial_state=state, verbose=False, yield_state=True, plot_every=1000000
        ):
            _old_flow_rate = _new_flow_rate
            _new_flow_rate = self.flow_rate(adjoint_state)
            adjoint_surface.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )

            adjoint_stress_averaged[current_time] = self.boundary_averaged_stress(
                adjoint_state
            )
            # with open("jacobian_25.dat", "a") as file:
            #     _str = str(adjoint_stress_averaged[current_time])
            #     file.write(_str)
            #     file.write("\n")
            current_time -= self.timer["dt"]

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0.0
        du[self.timer["T"]] = 0.0

        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())
        if grad_data["grad_norm"] is None:
            grad_data["grad_norm"] = (adjoint_stress_averaged * du).integrate()

        return np.array(du.values()[1:-1])


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

surface_model = SurfaceModel(nondim_constants, kappa_t0=0.05)
solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)
initial_state = (0.0, 0.0, 0.0)
initial_state = (dolf.Expression("0.1*x[0]*(1.0-x[0])", degree=2), 0.0, 0.0)


x0 = [0.0e-0 for _ in range(len(default_grid) - 2)]
top_bound = [0.06 for i in range(len(x0))]
low_bound = [-0.03 for i in range(len(x0))]
bnds = list(zip(low_bound, top_bound))
x0 = np.array(x0)


def calculate_taylor_remainders(solver, max_iter=30):
    energy = []
    grad_data["grad_norm"] = None
    state = solver._objective_state(x0)
    energy.append(solver._objective(state))
    grad = np.array(solver._jacobian(state))

    x = []
    y = []
    constant = 1.0e-1
    for i in range(1, max_iter):
        step_size = constant * 2.0 ** (-i)
        new_state = solver._objective_state(x0 + grad * step_size)
        energy.append(solver._objective(new_state))
        # remainders.append(energy[-1] - energy[0] - step_size * grad_data["grad_norm"])
        x.append(constant * 2.0 ** (-i))
        y.append(
            abs(energy[-1] - energy[0] - step_size * grad_data["grad_norm"]) / energy[0]
        )
    constant = 1.0e-6
    for i in range(1, max_iter):
        step_size = constant * 2.0 ** (-i)
        new_state = solver._objective_state(x0 + grad * step_size)
        energy.append(solver._objective(new_state))
        # remainders.append(energy[-1] - energy[0] - step_size * grad_data["grad_norm"])
        x.append(constant * 2.0 ** (-i))
        y.append(
            abs(energy[-1] - energy[0] - step_size * grad_data["grad_norm"]) / energy[0]
        )
    # for i, el in enumerate(remainders):
    #     x.append(constant * 2.0 ** (-i - 1))
    #     y.append(abs(el) / energy[0])
    return x[::-1], y[::-1]


run_taylor_test = True
if run_taylor_test:

    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(
            x,
            [x < x0],
            [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0],
        )

    # c = np.arange(0, 12)  # 4, 12, 2
    c = np.arange(100, 3320, 50)
    for iterations in c[::-1]:
        initial_resolution = 100 + iterations  # * 2 ** iterations  # iterations
        domain = IntervalDomain([p1, p2], resolution=initial_resolution)
        cut = 0.8
        # domain.mesh = resolve_mesh(domain.mesh, iterations=iterations, cut=cut)
        print("num cells: ", domain.mesh.num_cells())
        # domain.mesh = resolve_mesh(domain.mesh, 4, cut=1.0 - 0.1 * iterations)
        solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)

        x, y = calculate_taylor_remainders(solver, max_iter=14)
        with open("taylor_state_uniform_T_01_dt_0001.dat", "a") as file:
            # file.write(
            #     str((1.0 - cut) / (domain.mesh.num_cells() - initial_resolution * cut))
            #     + "\n"
            # )
            file.write(str(1.0 / domain.mesh.num_cells()) + "\n")
            # file.write(str(float(timer["dt"])) + "\n")
            file.write(",".join(map(str, x)))
            file.write("\n")
            file.write(",".join(map(str, y)))
            file.write("\n")

    exit()

solver.optimization_method = "TNC"
res = solver.minimize(x0, bnds)

plt.plot(res.x)
plt.show()
