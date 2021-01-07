from firecrest.api import LineElement, SimpleDomain, SpectralTVAcousticSolver


if __name__ == "__main__":
    control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    control_points_2 = [[1.0, 1.0], [0.0, 1.0]]
    boundary1 = LineElement(
        control_points_1, el_size=0.05, bcond={"noslip": True, "isothermal": True}
    )
    impedance = -0.1j
    force = 1.0 + 0.0j
    boundary2 = LineElement(
        control_points_2,
        el_size=0.05,
        bcond={"inhom_impedance": (impedance, force), "adiabatic": True},
        # bcond={"normal_force": 0.5 - 0.5j, "adiabatic": True},
        # bcond={"normal_velocity": 1.0, "adiabatic": True},
    )

    domain_boundaries = (boundary1, boundary2)
    domain = SimpleDomain(domain_boundaries)

    solver = SpectralTVAcousticSolver(domain, frequency=10.0j, Re=500.0, Pr=10.0)
    state = solver.solve()
    solver.output_field(state)

    real_mode, imag_mode = state[:3], state[3:]
    print(
        "stress: ",
        solver.forms.avg_normal_stress(real_mode, boundary2)
        + 1.0j * solver.forms.avg_normal_stress(imag_mode, boundary2),
    )
    print(
        "Zu + f: ",
        impedance
        * (
            solver.forms.mass_flow_rate(real_mode, boundary2)
            + 1.0j * solver.forms.mass_flow_rate(imag_mode, boundary2)
        )
        + force,
    )
