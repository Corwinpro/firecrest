from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
import dolfin as dolf
import math

current_time = 0.0


class SurfaceTensionBoundary:
    def __init__(self):
        self.tension_coefficient = 1.0e-3
        self.curvature = 1.0e-1

    def eval(self):
        return self.tension_coefficient * self.curvature

    def update_curvature(self, velocity):
        pass


control_points_1 = [[0.0, 0.0], [1.0e-16, 1.0]]
control_points_2 = [[1.0e-16, 1.0], [0.2, 1.0 - 1.0e-16]]
control_points_3 = [[0.2, 1.0 - 1.0e-16], [0.2, 1.0e-16]]
control_points_4 = [[0.2, 1.0e-16], [0.0, 0.0]]

boundary1 = LineElement(
    control_points_1, el_size=0.05, bcond={"noslip": True, "adiabatic": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.05, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=0.05, bcond={"noslip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=0.05,
    bcond={"normal_force": SurfaceTensionBoundary(), "adiabatic": True},
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)

solver = UnsteadyTVAcousticSolver(domain, Re=1.0e3, Pe=10.0, dt=1.0e-2)
initial_state = (
    dolf.Constant(0.0, cell=solver.domain.mesh.ufl_cell()),
    dolf.Constant((0.0, 0.0), cell=solver.domain.mesh.ufl_cell()),
    dolf.Constant(0.0, cell=solver.domain.mesh.ufl_cell()),
)

f = dolf.File("temp.pvd")
P = dolf.Function(solver.forms.pressure_function_space.collapse())
P.rename("P", "P")
P.interpolate(initial_state[0])
f << P

for i in range(100):
    old_state = initial_state
    w = solver.solve(initial_state)
    initial_state = w.split(True)
    initial_state[0].rename("P", "P")
    f << initial_state[0]
    current_time += solver._dt
    print(current_time)
