from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
import dolfin as dolf
from firecrest.misc.time_storage import TimeSeries
from decimal import Decimal


class NormalInflow:
    high = 1

    def __init__(self, value):
        self.counter = 0
        self.value = value

    def eval(self):
        if self.counter < NormalInflow.high:
            self.counter += 1
            return 0.0, self.value
        return 0.0, 0.0


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
    control_points_2, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=el_size, bcond={"noslip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=el_size / 2.0,
    bcond={"inflow": NormalInflow(1.0), "adiabatic": True},
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)


timer = {"dt": Decimal("0.01"), "T": Decimal("0.01")}
solver = UnsteadyTVAcousticSolver(domain, Re=5.0e3, Pr=10.0, timer=timer)
initial_state = (0.0, (0.0, 0.0), 0.0)
state = solver.solve_direct(initial_state, verbose=True)
final_state = state.last


print(
    "Final Energy: {}".format(
        dolf.assemble(solver.forms.temporal_component(final_state, final_state)) / 2.0
    )
)


final_state = list(final_state)
final_state[1] = -final_state[1]
final_state = tuple(final_state)

adjoint_history = solver.solve_adjoint(final_state, verbose=True)
print(adjoint_history)
adjoint_stress = adjoint_history.apply(lambda x: solver.forms.stress(x[0], x[1]))

adjoint_stress_averaged = adjoint_stress.apply(
    lambda x: dolf.assemble(
        dolf.dot(dolf.dot(x, solver.domain.n), solver.domain.n)
        * solver.domain.ds((boundary4.surface_index,))
    )
)

dU = 1.0e-3

dvelocity = TimeSeries.from_dict(
    {
        Decimal(i) * Decimal(timer["dt"]): dU
        for i in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)
dvelocity[0] = 0

delta_energy = (
    solver._dt
    * dU
    * (sum(adjoint_stress_averaged.values()) - 0.5 * adjoint_stress_averaged.first)
)
print(delta_energy)

denergy = (adjoint_stress_averaged * dvelocity).integrate()
print(denergy)

for i in range(1, 11):
    boundary4.bcond["inflow"] = NormalInflow(1.0 - i * dU)
    state = solver.solve_direct(initial_state, verbose=False)
    final_state = state.last
    print(
        "Final Energy: {}".format(
            dolf.assemble(solver.forms.temporal_component(final_state, final_state))
            / 2.0
        )
    )
