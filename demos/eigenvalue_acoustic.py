from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf


control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.05, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.05, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


solver = EigenvalueTVAcousticSolver(domain, complex_shift=0 + 2j, Re=100.0, Pe=1.0)
solver.solve()

ev, real_mode, imag_mode = solver.restore_eigenfunction(0)

file = dolf.File("mode.pvd")
file << real_mode[1], 0
file << imag_mode[1], 1

func = dolf.Function(solver.forms.function_space)
# dolf.assign(func.sub(0), real_mode[0]) # This works
dolf.assign(func, list(real_mode + imag_mode))
file << func.sub(4), 2
