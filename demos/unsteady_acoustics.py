from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
import dolfin as dolf


control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.05, bcond={"slip": True, "isothermal": True}  # noslip
)
boundary2 = LineElement(
    control_points_2, el_size=0.05, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)

solver = UnsteadyTVAcousticSolver(domain, Re=1.0e3, Pe=10.0, dt=1.0e-2)
initial_state = (
    dolf.Expression(
        "exp(-(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.5)*(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.5)/0.025)",
        # degree=2,
        element=solver.forms.pressure_function_space.ufl_element(),
    ),
    dolf.Constant((0.0, 0.0), cell=solver.domain.mesh.ufl_cell()),
    dolf.Constant(0.0, cell=solver.domain.mesh.ufl_cell()),
)

f = dolf.File("temp.pvd")
# P = dolf.Function(solver.forms.pressure_function_space.collapse())
# P.rename("P", "P")
# P.interpolate(initial_state[0])
# f << P
U = dolf.Function(solver.forms.velocity_function_space.collapse())
U.rename("u", "u")
U.interpolate(initial_state[1])
f << U

for i in range(100):
    old_state = initial_state
    w = solver.solve(initial_state)
    # dolf.assign(P, w.sub(0))
    # f << P
    initial_state = w.split(True)
    initial_state[1].rename("u", "u")
    f << initial_state[1]
    print(
        dolf.assemble(solver.forms.temporal_component(initial_state, initial_state)),
        dolf.assemble(solver.forms.spatial_component(initial_state, initial_state))
        * solver._dt,
    )
    # print(
    #     dolf.assemble(solver.forms.temporal_component(initial_state, initial_state))
    #     - dolf.assemble(solver.forms.temporal_component(initial_state, old_state))
    #     + dolf.assemble(solver.forms.spatial_component(initial_state, initial_state))
    #     * solver._dt
    # )
