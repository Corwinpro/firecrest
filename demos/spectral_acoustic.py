import dolfin as dolf

from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.spectral_tv_acoustic_solver import SpectralTVAcousticSolver

control_points_1 = [[1.0e-16, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
control_points_2 = [[1.0, 1.0], [1.0e-16, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.03, bcond={"noslip": True, "isothermal": True}
)
z = -0.1j
f = 1.0 + 0.0j
boundary2 = LineElement(
    # control_points_2, el_size=0.03, bcond={"normal_velocity": 1.0, "adiabatic": True}
    control_points_2,
    el_size=0.03,
    bcond={"inhom_impedance": (z, f), "adiabatic": True},
    # bcond={"normal_force": 0.5 - 0.5j, "adiabatic": True},
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


solver = SpectralTVAcousticSolver(domain, frequency=10.5j, Re=500.0, Pr=1.0)
state = solver.solve()
solver.output_field(state)


def normal_velocity(mode):
    return dolf.assemble(
        (dolf.dot(mode[1], solver.domain.n)) * solver.domain.ds(boundary2.surface_index)
    )


def _stress(mode):
    n = solver.domain.n
    return dolf.dot(dolf.dot(solver.forms.stress(mode[0], mode[1]), n), n)


def stress(mode):
    return dolf.assemble(_stress(mode) * solver.domain.ds(boundary2.surface_index))


real_mode = state[:3]
imag_mode = state[3:]
print("stress: ", stress(real_mode) + 1.0j * stress(imag_mode))
print(
    "zu + f: ", z * (normal_velocity(real_mode) + 1.0j * normal_velocity(imag_mode)) + f
)
