from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel
import dolfin as dolf
from collections import namedtuple

elsize = 0.08
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
mu = 1.0e-2
Re = rho * c_s * L / mu
a1 = 4.0 / 3.0
a2 = 8
kappa_prime = epsilon / nozzle_r ** 4 * 0.25  # up to a constant

# nondimensional
alpha_1 = nozzle_l / (3.1415 * nozzle_r ** 2) * a1
alpha_2 = nozzle_l / (3.1415 * nozzle_r ** 3) / Re * a2 / nozzle_r
gamma_nd = 2 * gamma_st / (rho * c_s ** 2 * nozzle_r * epsilon)


Constants = namedtuple(
    "Constants",
    [
        "density",
        "acoustic_mach",
        "channel_L",
        "nozzle_L",
        "nozzle_R",
        "surface_tension",
        "Re",
        "sound_speed",
    ],
)
printhead_constants = Constants(
    1.0e3, 1.0e-3, 100.0e-6, 10.0e-6, 10.0e-6, 50.0e-3, 5.0e3, 1.0e3
)

nondim_constants = Constants(
    1.0,
    printhead_constants.acoustic_mach,
    1,
    printhead_constants.nozzle_L / printhead_constants.channel_L,
    printhead_constants.nozzle_R / printhead_constants.channel_L,
    2.0
    * printhead_constants.surface_tension
    / (
        printhead_constants.density
        * printhead_constants.sound_speed ** 2.0
        * printhead_constants.nozzle_R
        * printhead_constants.acoustic_mach
    ),
    printhead_constants.Re,
    1.0,
)
current_time = 0.0


surface_model = SurfaceModel(nondim_constants, kappa_t0=0.25)

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
    el_size=elsize / 4.0,
    bcond={"noslip": True, "adiabatic": True},
)
boundary2 = LineElement(
    control_points_2,
    el_size=elsize / 4.0,
    bcond={"normal_force": surface_model, "adiabatic": True},
)
boundary_refine_right = LineElement(
    control_points_refine_right,
    el_size=elsize / 4.0,
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
    control_points_4, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
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


solver = UnsteadyTVAcousticSolver(domain, Re=5.0e3, Pr=10.0, dt=4.0e-3)
initial_state = (0.0, (0.0, 0.0), 0.0)


for i in range(10000):
    old_state = initial_state
    w = solver.solve(initial_state)

    initial_state = w.split(True)
    if i % 20 == 9:
        solver.output_field(initial_state)

    # Updating the curvature
    flow_rate = dolf.assemble(
        dolf.inner(initial_state[1], domain.n) * domain.ds((boundary2.surface_index,))
    )
    surface_model.update_curvature(flow_rate, solver._dt)

    current_time += solver._dt
    print(current_time, surface_model.kappa)
