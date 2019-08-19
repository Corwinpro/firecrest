from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf


control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.03, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.03, bcond={"impedance": 0.0, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


solver = EigenvalueTVAcousticSolver(domain, complex_shift=-0.3 + 3.0j, Re=500.0, Pr=1.0)
solver.solve()

mode_imag = dolf.File("mode_imag.pvd")
mode_real = dolf.File("mode_real.pvd")


def impedance_check(mode):
    return normal_velocity(mode) - dolf.assemble(
        -dolf.dot(
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
    imag_mode[1].rename("uI", "uI")
    real_mode[1].rename("uR", "uR")
    mode_real << real_mode[1], i
    mode_imag << imag_mode[1], i
    print("Z*un - sigma_n_n real: ", impedance_check(real_mode))
    print("Z*un - sigma_n_n imag: ", impedance_check(imag_mode))
    print("un real:", normal_velocity(real_mode))
    print("un imag:", normal_velocity(imag_mode))
