"""
Eingenvalue problem for a symmetric inkjet printhead geometry.

Two symmetric systems can be considered:
    - with slip boundary on the symmetry plane,
        This results in odd eigenmodes
    - with stress-free boundary on the symmetry plane.
        This results in even eigenmodes
"""
from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf
import matplotlib.pyplot as plt

control_points_1 = [[9.9, -0.2], [9.9, 0.0], [0.0, 0.0], [0.0, 3.0]]
control_points_2 = [[0.0, 3.0], [1.0, 3.0]]
control_points_3 = [[1.0, 3.0], [1.0, 1.0], [10.0, 1.0]]
control_points_4 = [[10.0, 1.0], [10.0, -0.2]]
control_points_5 = [[10.0, -0.2], [9.9, -0.2]]


el_size = 0.1
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
    control_points_4, el_size=el_size, bcond={"slip": True, "adiabatic": True}
)
boundary5 = LineElement(
    control_points_5, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4, boundary5)
domain = SimpleDomain(domain_boundaries)

solver = EigenvalueTVAcousticSolver(
    domain, complex_shift=-0.03 + 1.0j, Re=1000.0, Pe=1.0, nmodes=20
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

short_spectrum = [
    (-0.029796769962597662 + 0.35294042262432046j),
    (-0.029512514468998545 + 0.5581814336549202j),
    (-0.040852252316042476 + 0.15539180023941337j),
    (-0.03360330688324703 + 0.7822368881240528j),
]
plt.plot([ev.real for ev in short_spectrum], [ev.imag for ev in short_spectrum], "x")

plt.grid(True)
plt.show()
# file << real_mode[1], 0
# file << imag_mode[1], 1
