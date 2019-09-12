from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
import dolfin as dolf
from firecrest.misc.time_storage import TimeSeries
from decimal import Decimal

timer = {"dt": Decimal("0.001"), "T": Decimal("0.21")}


class NormalInflow:
    def __init__(self, series: TimeSeries):
        self.counter = 1
        self.series = list(series.values())

    def eval(self):
        if self.counter < len(self.series):
            value = 0.0, self.series[self.counter]
            self.counter += 1
            return value
        return 0.0, 0.0


def generate_inflow(amplitude):
    inflow = TimeSeries.from_dict(
        {
            Decimal(i) * Decimal(timer["dt"]): 0 if i == 0 else amplitude
            for i in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
        }
    )
    return inflow


inflow = generate_inflow(1.0)

length = 0.3
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
    control_points_2, el_size=el_size, bcond={"free": True, "adiabatic": True}
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


solver = UnsteadyTVAcousticSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)
initial_state = (0.0, (0.0, 0.0), 0.0)
state = next(solver.solve_direct(initial_state, verbose=True))
final_state = state.last


print(
    "Final Energy: {}".format(
        dolf.assemble(solver.forms.temporal_component(final_state, final_state)) / 2.0
    )
)


final_state = list(final_state)
final_state[1] = -final_state[1]
final_state = tuple(final_state)

adjoint_history = next(solver.solve_adjoint(final_state, verbose=True))

adjoint_stress = adjoint_history.apply(lambda x: solver.forms.stress(x[0], x[1]))
adjoint_stress_averaged = adjoint_stress.apply(
    lambda x: dolf.assemble(
        dolf.dot(dolf.dot(x, solver.domain.n), solver.domain.n)
        * solver.domain.ds((boundary4.surface_index,))
    )
)

dU = 1.0e-3
d_inflow = generate_inflow(dU)
delta_energy = (adjoint_stress_averaged * d_inflow).integrate()
print(delta_energy)

for i in range(1, 11):
    boundary4.bcond["inflow"] = NormalInflow(generate_inflow(1.0 - i * dU))
    state = next(solver.solve_direct(initial_state, verbose=False))
    final_state = state.last
    print(
        "Final Energy: {}".format(
            dolf.assemble(solver.forms.temporal_component(final_state, final_state))
            / 2.0
        )
    )
