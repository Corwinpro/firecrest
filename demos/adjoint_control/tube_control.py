from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.misc.time_storage import TimeSeries
from firecrest.misc.optimization_mixin import OptimizationMixin
import dolfin as dolf
from decimal import Decimal


timer = {"dt": Decimal("0.001"), "T": Decimal("0.01")}


class NormalInflow:
    def __init__(self, series: TimeSeries):
        self.counter = 1
        self.series = list(series.values())

    def eval(self):
        if self.counter < len(self.series):
            value = (0.0, -self.series[self.counter])
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


x0 = [1.0 for _ in range(len(default_grid) - 1)]
inflow = TimeSeries.from_list([0.0] + x0, default_grid)

length = 0.03
width = 0.05
offset = 0.005
control_points_1 = [[offset, 0.0], [0.0, 0.0], [1.0e-16, length]]
control_points_2 = [[1.0e-16, length], [width, length - 1.0e-16]]
control_points_3 = [[width, length - 1.0e-16], [width, 0.0], [width - offset, 1.0e-16]]
control_points_4 = [[width - offset, 1.0e-16], [offset, 0.0]]

el_size = 0.0025

boundary1 = LineElement(
    control_points_1, el_size=el_size, bcond={"noslip": True, "adiabatic": True}
)
boundary2 = LineElement(
    control_points_2, el_size=el_size, bcond={"noslip": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=el_size, bcond={"noslip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=el_size / 2.0,
    bcond={"inflow": NormalInflow(inflow), "adiabatic": True},
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)


class OptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _objective_state(self, control):
        boundary4.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list([0.0] + list(control) + [0.0], default_grid)
        )
        return self.solve_direct(initial_state, verbose=True).last

    def _objective(self, state):
        return self.forms.energy(state)

    def _jacobian(self, state):
        state = (state[0], -state[1], state[2])

        adjoint_history = self.solve_adjoint(state, verbose=True)
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

        print((adjoint_stress_averaged * du).integrate())
        return [d * self._dt for d in du.values()[1:-1]]


solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer, verbose=True)
initial_state = (0.0, (0.0, 0.0), 0.0)
x0 = [1.0 for _ in range(len(default_grid) - 2)]
bnds = tuple((-1.1, 1.1) for i in range(len(x0)))
res = solver.minimize(x0, bnds)
