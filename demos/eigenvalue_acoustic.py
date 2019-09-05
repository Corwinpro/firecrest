from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf

z = 2.0 + 1.5j

control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.02, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.02, bcond={"impedance": z, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


solver = EigenvalueTVAcousticSolver(domain, complex_shift=3.0j, Re=500.0, Pr=1.0)
solver.solve()


def stress(mode):
    return dolf.assemble(
        dolf.dot(
            dolf.dot(solver.forms.stress(mode[0], mode[1]), solver.domain.n),
            solver.domain.n,
        )
        * solver.domain.ds(boundary2.surface_index)
    )


def normal_velocity(mode):
    return dolf.assemble(
        (dolf.dot(mode[1], solver.domain.n)) * solver.domain.ds(boundary2.surface_index)
    )


for i in range(int(solver.nof_modes_converged / 2)):
    ev, real_mode, imag_mode = solver.extract_solution(i)

    solver.output_field(real_mode + imag_mode)

    z_u_real = z.real * normal_velocity(real_mode) - z.imag * normal_velocity(imag_mode)
    z_u_imag = z.real * normal_velocity(imag_mode) + z.imag * normal_velocity(real_mode)
    print("Z*u real: ", z_u_real)
    print("sigma_n_n real: ", stress(real_mode))
    print("difference: ", z_u_real - stress(real_mode))

    print("Z*u imag: ", z_u_imag)
    print("sigma_n_n imag: ", stress(imag_mode))
    print("difference: ", z_u_imag - stress(imag_mode))
