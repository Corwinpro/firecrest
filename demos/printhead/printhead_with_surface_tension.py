"""
Eingenvalue problem for a symmetric inkjet printhead geometry.

Nonlinear nozzle ROM boundary condition + iterative eigenvalue solver
"""
from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf
import matplotlib.pyplot as plt

initial_guess = -0.008216631505083012 + 0.17739548404541003j

s0 = initial_guess

Re = 6000.0

control_points_0 = [[9.1, -0.2], [9.1, 0.0]]
control_points_1 = [[9.1, 0.0], [0.0, 0.0], [0.0, 4.7]]
control_points_2 = [[0.0, 4.7], [2.0, 4.7]]
control_points_3 = [[2.0, 4.7], [2.0, 0.7], [9.2, 0.7]]
control_points_4 = [[9.2, 0.7], [9.2, -0.2]]
control_points_5 = [[9.2, -0.2], [9.1, -0.2]]


el_size = 0.06
boundary0 = LineElement(
    control_points_0, el_size=el_size / 4.0, bcond={"noslip": True, "isothermal": True}
)
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
    control_points_4,
    el_size=el_size / 2.0,
    bcond={"slip": True, "adiabatic": True},  # symmetric
    # bcond={"free": True, "isothermal": True},  # antisymmetric
)
gamma = -1.0
default_capacitance = 1.0e3 * 0.1 / 4.0 / gamma
boundary5 = LineElement(
    control_points_5,
    el_size=el_size / 8.0,
    # bcond={"free": True, "adiabatic": True},
    bcond={
        "nozzle_impedance": {
            "capacitance": default_capacitance,
            "inductance": 0.1,
            "resistance": 0.1 / Re,
            "frequency": s0,
        },
        "adiabatic": True,
    },
)

domain_boundaries = (boundary0, boundary1, boundary2, boundary3, boundary4, boundary5)
domain = SimpleDomain(domain_boundaries)


def normal_velocity(mode):
    return dolf.assemble(
        (dolf.dot(mode[1], solver.domain.n)) * solver.domain.ds(boundary5.surface_index)
    )


def _stress(mode):
    n = solver.domain.n
    return dolf.dot(dolf.dot(solver.forms.stress(mode[0], mode[1]), n), n)


def stress(mode):
    return dolf.assemble(_stress(mode) * solver.domain.ds(boundary5.surface_index))


spectrum = [s0]
f_spectrum = []

solver = EigenvalueTVAcousticSolver(domain, complex_shift=s0, Re=Re, Pe=1.0, nmodes=2)
solver.solve()

s_old = s0
s0, real_mode, imag_mode = solver.extract_solution(0)
solver.output_field(real_mode + imag_mode)

ds = s_old - s0
print("abs|s-s0|", abs(ds))

spectrum.append(s0)
f_spectrum.append(s0)

boundary5.bcond["nozzle_impedance"]["frequency"] = s0

for i in range(10):

    solver = EigenvalueTVAcousticSolver(
        domain, complex_shift=s0, Re=Re, Pe=1.0, nmodes=2
    )
    solver.solve()
    s_old = s0
    s0, real_mode, imag_mode = solver.extract_solution(0)

    f_spectrum.append(s0)
    f_prime = (f_spectrum[-1] - f_spectrum[-2]) / (spectrum[-1] - spectrum[-2])
    alpha = 1.0 / (1 - f_prime)
    s0 = alpha * f_spectrum[-1] + (1.0 - alpha) * spectrum[-1]
    spectrum.append(s0)

    ds = abs(spectrum[-1] - spectrum[-2])
    print("abs|s-s0|", ds)

    boundary5.bcond["nozzle_impedance"]["frequency"] = spectrum[-1]
    solver.output_field(real_mode + imag_mode)

    if abs(ds) < 1.0e-9:
        break


plt.plot([ev.real for ev in spectrum], [ev.imag for ev in spectrum], "o-")

plt.grid(True)
plt.show()
