from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf


control_points_1 = [[9.75, -0.2], [9.75, 0.0], [0.0, 0.0], [0.0, 3.0]]
control_points_2 = [[0.0, 3.0], [1.0, 3.0]]
control_points_3 = [[1.0, 3.0], [1.0, 1.0], [10.0, 1.0]]
control_points_4 = [[10.0, 1.0], [10.0, -0.2], [9.75, -0.2]]

boundary1 = LineElement(
    control_points_1, el_size=0.05, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.05, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=0.05, bcond={"noslip": True, "isothermal": True}
)
boundary4 = LineElement(
    control_points_4, el_size=0.05, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)

solver = EigenvalueTVAcousticSolver(
    domain, complex_shift=0.0 + 0.25j, Re=100.0, Pe=1.0, nmodes=10
)
solver.solve()

file = dolf.File("mode.pvd")
for i in range(solver.nof_modes_converged):
    ev, real_mode, imag_mode = solver.restore_eigenfunction(i)
    real_mode[1].rename("u", "u")
    file << real_mode[1], i

# file << real_mode[1], 0
# file << imag_mode[1], 1
