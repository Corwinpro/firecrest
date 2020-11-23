from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.spectral_tv_acoustic_solver import SpectralTVAcousticSolver
import dolfin as dolf
import matplotlib.pyplot as plt
import numpy as np
import cmath

nozzle_height = -0.2
control_points_0 = [[9.1, nozzle_height], [9.1, 0.0]]
control_points_1 = [[9.1, 0.0], [0.0, 0.0], [0.0, 4.7]]
control_points_2 = [[0.0, 4.7], [2.0, 4.7]]
control_points_3 = [[2.0, 4.7], [2.0, 0.7], [5.2, 0.7]]
control_points_3a = [[5.2, 0.7], [9.2, 0.7]]
control_points_4 = [[9.2, 0.7], [9.2, nozzle_height]]
control_points_5 = [[9.2, nozzle_height], [9.1, nozzle_height]]


el_size = 0.15
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
boundary3a = LineElement(
    control_points_3a,
    el_size=el_size,
    bcond={"normal_velocity": 1.0, "isothermal": True},
)
boundary4 = LineElement(
    control_points_4,
    el_size=el_size / 2.0,
    bcond={"slip": True, "adiabatic": True},  # free or slip
)
boundary5 = LineElement(
    control_points_5, el_size=el_size / 8.0, bcond={"impedance": 0.0, "adiabatic": True}
)

domain_boundaries = (
    boundary0,
    boundary1,
    boundary2,
    boundary3,
    boundary3a,
    boundary4,
    boundary5,
)
domain = SimpleDomain(domain_boundaries)


def normal_velocity(mode, solver):
    return dolf.assemble(
        (dolf.dot(mode[1], solver.domain.n)) * solver.domain.ds(boundary5.surface_index)
    )


def _stress(mode, solver):
    n = solver.domain.n
    return dolf.dot(dolf.dot(solver.forms.stress(mode[0], mode[1]), n), n)


def stress(mode, solver):
    return dolf.assemble(
        _stress(mode, solver) * solver.domain.ds(boundary5.surface_index)
    )


def impedance(frequency):
    # return -1.0j / frequency * 0.1 * 1.0e-3 - 1.0j * frequency * 0.1 + 1.0 / 500.0
    return -4.0 * 0.1 / (frequency * 1.0j) - 0.1j * frequency + 0.1 / 6000.0 / 0.01


def standard_impedance():

    flux = []
    flux_free = []

    for frequency in np.logspace(-4, 3, 50):
        solver = SpectralTVAcousticSolver(
            domain, frequency=frequency, Re=6000.0, Pr=1.0
        )
        boundary5.bcond["impedance"] = impedance(frequency)
        state = solver.solve()
        _flux = (
            normal_velocity(state[:3], solver)
            + normal_velocity(state[3:], solver) * 1.0j
        )
        print(frequency, _flux)
        flux.append((frequency, abs(_flux), cmath.phase(_flux)))

        boundary5.bcond["impedance"] = 0.0
        state = solver.solve()
        _flux = (
            normal_velocity(state[:3], solver)
            + normal_velocity(state[3:], solver) * 1.0j
        )
        flux_free.append((frequency, abs(_flux), cmath.phase(_flux)))

    print(flux)
    print(flux_free)

    flux = np.array(flux)

    flux_free = np.array(flux_free)

    return flux, flux_free


def nozzle_impedance():
    flux = []

    for frequency in np.logspace(-4, 2, 30):
        # Solving the first part
        boundary3a.bcond["normal_velocity"] = 1.0
        # boundary5.bcond = {"impedance": -4.0 / (1.0j * frequency), "adiabatic": True}
        boundary5.bcond = {
            "inhom_nozzle_impedance": {
                "frequency": frequency,
                "inductance": 0.1,
                "resistance": 0.1 / 6000.0,
                "force": 0.0,
            },
            "adiabatic": True,
        }
        solver = SpectralTVAcousticSolver(
            domain, frequency=frequency, Re=6000.0, Pr=1.0
        )
        state_1 = solver.solve()
        mass_flux_1 = (
            normal_velocity(state_1[:3], solver)
            + normal_velocity(state_1[3:], solver) * 1.0j
        )
        # solver.output_field(state_1)

        boundary3a.bcond["normal_velocity"] = 0.0
        boundary5.bcond = {
            "inhom_nozzle_impedance": {
                "frequency": frequency,
                "inductance": 0.1,
                "resistance": 0.1 / 6000.0,
                "force": 1.0,
            },
            "adiabatic": True,
        }
        solver = SpectralTVAcousticSolver(
            domain, frequency=frequency, Re=6000.0, Pr=1.0
        )
        state_2 = solver.solve()
        mass_flux_2 = (
            normal_velocity(state_2[:3], solver)
            + normal_velocity(state_2[3:], solver) * 1.0j
        )
        # solver.output_field(state_1)
        print(mass_flux_1, mass_flux_2)

        mass_flux = mass_flux_1 / (1 - (-1.0) / (1.0j * frequency) * mass_flux_2)
        flux.append((frequency, abs(mass_flux), cmath.phase(mass_flux)))

    print(flux)

    flux = np.array(flux)
    return flux


#
# flux, flux_free = standard_impedance()
#
# plt.loglog(flux[:, 0], flux[:, 1], "o-")
# plt.loglog(flux_free[:, 0], flux_free[:, 1], "o-")


impedance_flux = nozzle_impedance()

plt.loglog(impedance_flux[:, 0], impedance_flux[:, 1], "o-")

plt.show()
