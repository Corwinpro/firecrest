from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf
import matplotlib.pyplot as plt

control_points_1 = [[9.9, -0.2], [9.9, 0.0], [0.0, 0.0], [0.0, 3.0]]
control_points_2 = [[0.0, 3.0], [1.0, 3.0]]
control_points_3 = [[1.0, 3.0], [1.0, 1.0], [19.0, 1.0], [19.0, 3.0]]
control_points_4 = [[19.0, 3.0], [20.0, 3.0]]
control_points_5 = [[20.0, 3.0], [20.0, 0.0], [10.1, 0.0], [10.1, -0.2]]
control_points_6 = [[10.1, -0.2], [9.9, -0.2]]

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
    control_points_4, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
boundary5 = LineElement(
    control_points_5, el_size=el_size, bcond={"noslip": True, "isothermal": True}
)
boundary6 = LineElement(
    control_points_6, el_size=el_size, bcond={"free": True, "adiabatic": True}
)

domain_boundaries = (boundary1, boundary2, boundary3, boundary4, boundary5, boundary6)
domain = SimpleDomain(domain_boundaries)

solver = EigenvalueTVAcousticSolver(
    domain, complex_shift=-0.03 + 1.j, Re=1000.0, Pe=1.0, nmodes=30
)
solver.solve()

spectrum = []
mode_imag = dolf.File("long_mode_imag.pvd")
mode_real = dolf.File("long_mode_real.pvd")

for i in range(int(solver.nof_modes_converged / 2)):
    ev, real_mode, imag_mode = solver.extract_solution(i)
    spectrum.append(ev)
    imag_mode[1].rename("uI", "uI")
    real_mode[1].rename("uR", "uR")
    mode_real << real_mode[1], i
    mode_imag << imag_mode[1], i


plt.plot([ev.real for ev in spectrum], [ev.imag for ev in spectrum], "o")
plt.grid(True)

free_spectrum = [
    (-0.01837855770075929 + 0.42820438785413684j),
    (-0.01267319444377846 + 0.21589516829157068j),
    (-0.024611285323162313 + 0.647112099304535j),
]
slip_spectrum = [
    (-0.029796769962599633 + 0.3529404226243147j),
    (-0.02951251446888157 + 0.558181433654887j),
    (-0.040852252322724575 + 0.15539180023453208j),
    (-0.033603306964518055 + 0.7822368882457811j),
]
plt.plot([ev.real for ev in free_spectrum], [ev.imag for ev in free_spectrum], "x")
plt.plot([ev.real for ev in slip_spectrum], [ev.imag for ev in slip_spectrum], "*", color='k')

plt.show()
# file << real_mode[1], 0
# file << imag_mode[1], 1
