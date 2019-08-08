from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel

import dolfin as dolf

current_time = 0.0


control_points_1 = [[0.0, 0.0], [1.0e-16, 1.0]]
control_points_2 = [[1.0e-16, 1.0], [0.2, 1.0 - 1.0e-16]]
control_points_3 = [[0.2, 1.0 - 1.0e-16], [0.2, 1.0e-16]]
control_points_4 = [[0.2, 1.0e-16], [0.0, 0.0]]

el_size = 0.01

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
    el_size=el_size,
    bcond={"inflow": (0., 1.), "adiabatic": True},
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)

solver = UnsteadyTVAcousticSolver(domain, Re=5.0e3, Pr=10.0, dt=1.0e-3)
initial_state = (0.0, (0.0, 0.0), 0.0)


for i in range(1000):
    old_state = initial_state
    w = solver.solve(initial_state)

    initial_state = w.split(True)
    if i % 10 == 9:
        solver.output_field(initial_state)

    current_time += solver._dt
    print(current_time)
