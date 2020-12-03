import dolfin as dolf

from firecrest.api import SimpleDomain, LineElement, EigenvalueTVAcousticSolver

z = -1.0 - 2.0j

control_points_1 = [
    [0.0, 1.0],
    [0.0, 0.8],
    [0.5, 0.8],
    [0.5, 0.7],
    [0.0, 0.7],
    [0.0, 0.0],
    [1.0, 0.0],
]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.02, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.02, bcond={"impedance": z, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


if __name__ == "__main__":

    solver = EigenvalueTVAcousticSolver(domain, complex_shift=-3.0j, Re=500.0, Pr=1.0)
    (ev, real_mode, imag_mode), = solver.solve(number_of_modes=1)

    solver.output_field(real_mode + imag_mode)
    energy = dolf.assemble(
        solver.forms.temporal_component(real_mode, real_mode)
    ) + dolf.assemble(solver.forms.temporal_component(imag_mode, imag_mode))
    print(
        "Estimated eigenvalue: ",
        (
            -dolf.assemble(solver.forms.spatial_component(real_mode, real_mode))
            - dolf.assemble(solver.forms.spatial_component(imag_mode, imag_mode))
            + z.real
            * dolf.assemble(
                (
                    dolf.inner(real_mode[1], real_mode[1])
                    + dolf.inner(imag_mode[1], imag_mode[1])
                )
                * domain.ds((boundary2.surface_index,))
            )
        )
        / energy
        + dolf.assemble(
            solver.forms.spatial_component(real_mode, imag_mode)
            - solver.forms.spatial_component(imag_mode, real_mode)
            + z.imag
            * (
                dolf.inner(real_mode[1], real_mode[1])
                + dolf.inner(imag_mode[1], imag_mode[1])
            )
            * domain.ds(boundary2.surface_index)
        )
        / energy
        * 1.0j,
    )

    z_u_real = z.real * solver.forms.mass_flow_rate(
        real_mode, boundary2
    ) - z.imag * solver.forms.mass_flow_rate(imag_mode, boundary2)
    z_u_imag = z.real * solver.forms.mass_flow_rate(
        imag_mode, boundary2
    ) + z.imag * solver.forms.mass_flow_rate(real_mode, boundary2)
    print("Z*u real: ", z_u_real)
    real_stress = solver.forms.avg_normal_stress(real_mode, boundary2)
    print("sigma_n_n real: ", real_stress)
    print("difference: ", z_u_real - real_stress)

    print("Z*u imag: ", z_u_imag)
    imaginary_stress = solver.forms.avg_normal_stress(imag_mode, boundary2)
    print("sigma_n_n imag: ", imaginary_stress)
    print("difference: ", z_u_imag - imaginary_stress)
