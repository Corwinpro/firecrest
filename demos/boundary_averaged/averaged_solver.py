from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.spectral_tv_acoustic_solver import SpectralTVAcousticSolver
import dolfin as dolf
import matplotlib.pyplot as plt
import numpy as np

elsize = 0.08

# Channel dimensions
height = 0.7
length = 10.0
offset_top = 1.0
nozzle_r = 0.1
nozzle_l = 0.2
nozzle_offset = 0.2

# Define constants
L = 1.0e-4
c_s = 1.0e3
rho = 1.0e3
epsilon = 1.0e-3
gamma_st = 50.0e-3
mu = 1.0e-3  # 2
Re = rho * c_s * L / mu
a1 = 4.0 / 3.0
a2 = 8
kappa_prime = epsilon / nozzle_r ** 4 * 0.25  # up to a constant

# nondimensional
alpha_1 = nozzle_l / (3.1415 * nozzle_r ** 2) * a1
alpha_2 = nozzle_l / (3.1415 * nozzle_r ** 3) / Re * a2 / nozzle_r
gamma_nd = 2 * gamma_st / (rho * c_s ** 2 * nozzle_r * epsilon)


def bc_alpha(nondim_frequency):
    return (
        -gamma_nd * kappa_prime / nondim_frequency
        - alpha_1 * nondim_frequency
        - alpha_2
    )


def hz_to_npfreq(dimensional_frequency):
    return dimensional_frequency * 1.0j * L / c_s


plot_impedance = False
if plot_impedance:
    frequency_range = np.logspace(1, 7, 100)
    plt.loglog(
        [frequency for frequency in frequency_range],
        [abs(bc_alpha(hz_to_npfreq(frequency))) for frequency in frequency_range],
        "o",
        color="b",
    )
    plt.show()
    exit()

omega = hz_to_npfreq(1.0e5)
alpha = bc_alpha(omega)
print(alpha)

control_points_1 = [[offset_top, height], [1.0e-16, height]]
control_points_free_left = [[1.0e-16, height], [0.0, 0.0 + 1.0e-16]]
control_points_bot_left = [
    [0.0, 0.0 + 1.0e-16],
    [length / 2.0 - nozzle_r - nozzle_offset, 0.0],
]
control_points_refine_left = [
    [length / 2.0 - nozzle_r - nozzle_offset, 0.0],
    [length / 2.0 - nozzle_r, 0.0],
    [length / 2.0 - nozzle_r, -nozzle_l],
]
control_points_2 = [
    [length / 2.0 - nozzle_r, -nozzle_l],
    [length / 2.0 + nozzle_r, -nozzle_l + 1.0e-16],
]
control_points_refine_right = [
    [length / 2.0 + nozzle_r, -nozzle_l + 1.0e-16],
    [length / 2.0 + nozzle_r, 0.0],
    [length / 2.0 + nozzle_r + nozzle_offset, 0.0],
]
control_points_bot_right = [
    [length / 2.0 + nozzle_r + nozzle_offset, 0.0],
    [length, 0.0 + 1.0e-16],
]
control_points_free_right = [[length, 0.0 + 1.0e-16], [length, height]]
control_points_3 = [[length, height], [length - offset_top, height + 1.0e-16]]
control_points_4 = [[length - offset_top, height + 1.0e-16], [offset_top, height]]


boundary1 = LineElement(
    control_points_1, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_free_left = LineElement(
    control_points_free_left, el_size=elsize, bcond={"free": True, "adiabatic": True}
)
boundary_bot_left = LineElement(
    control_points_bot_left, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_refine_left = LineElement(
    control_points_refine_left,
    el_size=elsize * 0.1,
    bcond={"noslip": True, "adiabatic": True},
)
fR = 0.02947897387857517
fI = 0.04766058826396886
boundary2 = LineElement(
    control_points_2,
    el_size=elsize * 0.1,
    bcond={
        "shape_impedance": alpha,
        # "avg_velocity": alpha,
        # "free": True,
        # "normal_force": 0.5 * (fR + fI) - 0.5j * (fR - fI),
        "adiabatic": True,
    },
)
boundary2.velocity_shape = dolf.Expression(
    "3./4./r/r/r*(x[0]-L+r)*(L+r-x[0])", r=nozzle_r, L=length / 2.0, degree=2
)
boundary_refine_right = LineElement(
    control_points_refine_right,
    el_size=elsize * 0.1,
    bcond={"noslip": True, "adiabatic": True},
)
boundary_bot_right = LineElement(
    control_points_bot_right, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_free_right = LineElement(
    control_points_free_right, el_size=elsize, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=elsize,
    bcond={"inflow": (0.0, 1.0 + 1.0j), "adiabatic": True},
)
domain_boundaries = (
    boundary1,
    boundary_free_left,
    boundary_bot_left,
    boundary_refine_left,
    boundary2,
    boundary_refine_right,
    boundary_bot_right,
    boundary_free_right,
    boundary3,
    boundary4,
)
domain = SimpleDomain(domain_boundaries)


def stress(mode, boundary=boundary2):
    return dolf.assemble(
        dolf.dot(
            dolf.dot(solver.forms.stress(mode[0], mode[1]), solver.domain.n),
            solver.domain.n,
        )
        * solver.domain.ds(boundary.surface_index)
    ) / (2 * nozzle_r)


def normal_velocity(mode, boundary=boundary2):
    return dolf.assemble(
        (dolf.dot(mode[1], solver.domain.n)) * solver.domain.ds(boundary.surface_index)
    )


#
# with open("uavg.txt", "w") as f:
#     for i in np.logspace(3, 5, 20):
#         omega = hz_to_npfreq(i)
#         alpha = bc_alpha(omega)
#         print("frequency. Hz:", i)
#         boundary2.bcond["avg_velocity"] = alpha
#
#         solver = SpectralTVAcousticSolver(domain, frequency=omega, Re=Re, Pr=1.0)
#         state = solver.solve_petsc()
#         real_mode = state[:3]
#         imag_mode = state[3:]
#         # uavg_real = alpha.real * normal_velocity(
#         #     real_mode
#         # ) - alpha.imag * normal_velocity(imag_mode)
#         # uavg_imag = alpha.real * normal_velocity(
#         #     imag_mode
#         # ) + alpha.imag * normal_velocity(real_mode)
#         x = i
#         plt.plot(x, normal_velocity(real_mode), "o", c="b")
#         plt.plot(x, normal_velocity(imag_mode), "*", c="r")
#         # plt.plot(x, (uavg_imag ** 2.0 + uavg_real ** 2.0) ** 0.5, "x", c="g")
#         # plt.plot(x, 10 * stress(real_mode), "o", c="g")
#         # plt.plot(x, 10 * stress(imag_mode), "*", c="c")
#         f.write(
#             str(
#                 [
#                     i,
#                     normal_velocity(real_mode),
#                     normal_velocity(imag_mode),
#                     stress(real_mode),
#                     stress(imag_mode),
#                 ]
#             )
#         )
#         f.write("\n")
# plt.show()
# exit()

solver = SpectralTVAcousticSolver(domain, frequency=omega, Re=Re, Pr=1.0)
state = solver.solve()
solver.output_field(state)

real_mode = state[:3]
imag_mode = state[3:]

uavg_real = alpha.real * normal_velocity(real_mode) - alpha.imag * normal_velocity(
    imag_mode
)
uavg_imag = alpha.real * normal_velocity(imag_mode) + alpha.imag * normal_velocity(
    real_mode
)

print("UavgR: ", normal_velocity(real_mode))
print("UavgI: ", normal_velocity(imag_mode), "\n")

print("Z*u real: ", uavg_real)
print("sigma_n_n real: ", stress(real_mode))
print("difference: ", uavg_real - stress(real_mode), "\n")

print("Z*u imag: ", uavg_imag)
print("sigma_n_n imag: ", stress(imag_mode))
print("difference: ", uavg_imag - stress(imag_mode))


f = dolf.File("Visualization/stress_normal.pvd")
st = dolf.project(
    dolf.dot(
        dolf.dot(
            solver.forms.stress(imag_mode[0], imag_mode[1]), dolf.as_vector((0.0, 1.0))
        ),
        dolf.as_vector((0.0, 1.0)),
    ),
    dolf.FunctionSpace(solver.domain.mesh, "CG", 2),
)
st.rename("sigma_n", "sigma_n")
f << st

f = dolf.File("Visualization/stress_tangent.pvd")
st = dolf.project(
    dolf.dot(
        dolf.dot(
            solver.forms.stress(imag_mode[0], imag_mode[1]), dolf.as_vector((0.0, 1.0))
        ),
        dolf.as_vector((1.0, 0.0)),
    ),
    dolf.FunctionSpace(solver.domain.mesh, "CG", 2),
)
st.rename("sigma_t", "sigma_t")
f << st
