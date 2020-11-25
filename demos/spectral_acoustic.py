import dolfin as dolf

from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.spectral_tv_acoustic_solver import SpectralTVAcousticSolver


def mass_flux(mode, domain, surface_index):
    return dolf.assemble((dolf.dot(mode[1], domain.n)) * domain.ds(surface_index))


def stress(mode, domain, forms, surface_index):
    _stress = dolf.dot(dolf.dot(forms.stress(mode[0], mode[1]), domain.n), domain.n)
    return dolf.assemble(_stress * domain.ds(surface_index))


if __name__ == "__main__":
    control_points_1 = [[1.0e-16, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    control_points_2 = [[1.0, 1.0], [1.0e-16, 1.0]]
    boundary1 = LineElement(
        control_points_1, el_size=0.01, bcond={"noslip": True, "isothermal": True}
    )
    impedance = -0.1j
    force = 1.0 + 0.0j
    boundary2 = LineElement(
        control_points_2,
        el_size=0.03,
        bcond={"inhom_impedance": (impedance, force), "adiabatic": True},
        # bcond={"normal_force": 0.5 - 0.5j, "adiabatic": True},
        # bcond={"normal_velocity": 1.0, "adiabatic": True},
    )

    domain_boundaries = (boundary1, boundary2)
    domain = SimpleDomain(domain_boundaries)

    solver = SpectralTVAcousticSolver(domain, frequency=1.0j, Re=500.0, Pr=10.0)
    state = solver.solve()
    solver.output_field(state)

    real_mode, imag_mode = state[:3], state[3:]
    print(
        "stress: ",
        stress(real_mode, domain, solver.forms, boundary2.surface_index)
        + 1.0j * stress(imag_mode, domain, solver.forms, boundary2.surface_index),
    )
    print(
        "Zu + f: ",
        impedance
        * (
            mass_flux(real_mode, domain=domain, surface_index=2)
            + 1.0j * mass_flux(imag_mode, domain=domain, surface_index=2)
        )
        + force,
    )
