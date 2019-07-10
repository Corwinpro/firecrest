from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf
import matplotlib.pyplot as plt

control_points_1 = [[9.75, -0.2], [9.75, 0.0], [0.0, 0.0], [0.0, 3.0]]
control_points_2 = [[0.0, 3.0], [1.0, 3.0]]
control_points_3 = [[1.0, 3.0], [1.0, 1.0], [10.0, 1.0]]
control_points_4 = [[10.0, 1.0], [10.0, -0.2], [9.75, -0.2]]

el_size = 0.05
boundary1 = LineElement(
    control_points_1, el_size=el_size, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=el_size, bcond={"noslip": True, "isothermal": True}
)
boundary4 = LineElement(
    control_points_4, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)

solver = EigenvalueTVAcousticSolver(
    domain, complex_shift=-0.02 + 0.5j, Re=1000.0, Pe=1.0, nmodes=10
)
solver.solve()

spectrum = []
mode_imag = dolf.File("mode_imag.pvd")
mode_real = dolf.File("mode_real.pvd")

for i in range(int(solver.nof_modes_converged / 2)):
    ev, real_mode, imag_mode = solver.extract_solution(i)
    spectrum.append(ev)
    imag_mode[1].rename("uI", "uI")
    real_mode[1].rename("uR", "uR")
    mode_real << real_mode[1], i
    mode_imag << imag_mode[1], i


plt.plot([ev.real for ev in spectrum], [ev.imag for ev in spectrum], "o")
plt.grid(True)
plt.show()
# file << real_mode[1], 0
# file << imag_mode[1], 1
