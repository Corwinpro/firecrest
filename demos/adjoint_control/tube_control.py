from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.misc.time_storage import TimeSeries
from firecrest.misc.optimization_mixin import OptimizationMixin
from firecrest.models.free_surface_cap import SurfaceModel, AdjointSurfaceModel
import dolfin as dolf
from decimal import Decimal
import numpy as np

timer = {"dt": Decimal("0.001"), "T": Decimal("0.002")}


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

length = 0.01
width = 0.025
offset = 0.005
control_points_1 = [[0.0, 0.0], [1.0e-16, length]]
control_points_2 = [[1.0e-16, length], [width, length - 1.0e-16]]
control_points_3 = [[width, length - 1.0e-16], [width, 1.0e-16]]
control_points_4 = [[width, 1.0e-16], [0.0, 0.0]]

el_size = 0.0001

boundary1 = LineElement(
    control_points_1, el_size=el_size, bcond={"slip": True, "adiabatic": True}
)
boundary2 = LineElement(
    control_points_2, el_size=el_size, bcond={"inflow": (0.0, 0.0), "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=el_size, bcond={"slip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4, el_size=el_size, bcond={"normal_force": 0.0, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)


class OptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _objective_state(self, control):
        boundary2.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list([0.0] + list(control) + [0.0], default_grid)
        )
        return next(self.solve_direct(initial_state, verbose=False)).last

    def _objective(self, state):
        return self.forms.energy(state)

    def _jacobian(self, state):
        state = (state[0], -state[1], state[2])

        adjoint_history = next(self.solve_adjoint(state, verbose=False))
        adjoint_stress = adjoint_history.apply(lambda x: self.forms.stress(x[0], x[1]))
        adjoint_stress_averaged = adjoint_stress.apply(
            lambda x: dolf.assemble(
                dolf.dot(dolf.dot(x, self.domain.n), self.domain.n)
                * self.domain.ds((boundary2.surface_index,))
            )
        )

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0
        du[timer["T"]] = 0

        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())

        return du.values()[1:-1]


class Constants:
    acoustic_mach = 1.0e-3
    surface_tension = 1.0e-2
    nozzle_R = width / 2.0


nondim_constants = Constants()


class OptimizationSolverFreeSurface(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flow_rate(self, state):
        return dolf.assemble(
            dolf.inner(state[1], self.domain.n)
            * self.domain.ds((boundary4.surface_index,))
        )

    def _objective_state(self, control):
        boundary2.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list([0.0] + list(control) + [0.0], default_grid)
        )

        surface_model = SurfaceModel(nondim_constants, kappa_t0=0.25)
        boundary4.bcond["normal_force"] = surface_model

        _old_flow_rate = 0
        for state in self.solve_direct(initial_state, verbose=False, yield_state=True):
            if isinstance(state, TimeSeries):
                direct_history = state
                break
            # print(
            #     "Surface energy: ",
            #     surface_model.surface_energy(surface_model.kappa),
            #     "kappa: ",
            #     surface_model.kappa,
            #     "Energy flux: ",
            #     dolf.assemble(
            #         dolf.dot(
            #             dolf.dot(self.forms.stress(state[0], state[1]), self.domain.n),
            #             state[1],
            #         )
            #         * self.domain.ds(boundary4.surface_index)
            #     ),
            # )
            # print(
            #     "Avg stress:",
            #     dolf.assemble(
            #         dolf.dot(
            #             dolf.dot(self.forms.stress(state[0], state[1]), self.domain.n),
            #             self.domain.n,
            #         )
            #         * self.domain.ds(boundary4.surface_index)
            #     )
            #     / width,
            #     "Pressure from free surface:",
            #     surface_model.eval(),
            # )

            _new_flow_rate = self.flow_rate(state)
            surface_model.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )
            _old_flow_rate = _new_flow_rate

        return direct_history.last, surface_model

    def _objective(self, state, verbose=True):
        return self.forms.energy(state[0]) + state[1].surface_energy(state[1].kappa)

    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        boundary4.bcond["normal_force"] = adjoint_surface

        _old_flow_rate = 0
        _new_flow_rate = 0
        adjoint_surface.update_curvature(
            0.5 * (_new_flow_rate + _old_flow_rate), self._dt
        )
        for adjoint_state in self.solve_adjoint(
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
                * self.domain.ds((boundary2.surface_index,))
            )
        )

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0.0
        du[timer["T"]] = 0.0
        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())

        return du.values()[1:-1]


solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer, verbose=False)
# initial_state = (dolf.Expression("x[1]*(0.1-x[1])*500", degree=2), (0.0, 0.0), 0.0)

# solver = OptimizationSolverFreeSurface(
#     domain, Re=5.0e3, Pr=10.0, timer=timer, verbose=False
# )
initial_state = (0.0, (0.0, 0.0), 0.0)
# initial_state = (
#     dolf.Expression(
#         "x[1] < eps ? p : 0",
#         p=SurfaceModel(nondim_constants, kappa_t0=0.25).pressure,
#         eps=2.0e-3,
#         degree=2,
#     ),
#     (0.0, 0.0),
#     0.0,
# )
# initial_state = (dolf.Expression("x[1]*(0.2-x[1])*50", degree=2), (0.0, 0.0), 0.0)

x0 = [0.01 for _ in range(len(default_grid) - 2)]
# bnds = tuple((-1.1, 1.1) for i in range(len(x0)))
# res = solver.minimize(x0, bnds)


energy = []
state = solver._objective_state(x0)
energy.append(solver._objective(state))
print(energy[-1])
grad = np.array(solver._jacobian(state))

for i in range(3, 34, 3):
    new_state = solver._objective_state(np.array(x0) + grad * 1.0e-4 * i)
    energy.append(solver._objective(new_state))
    print(i, energy[-1])
