from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel

import dolfin as dolf
from collections import namedtuple

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


surface_model = SurfaceModel(nondim_constants)

control_points_1 = [[0.0, 0.0], [1.0e-16, 1.0]]
control_points_2 = [[1.0e-16, 1.0], [0.2, 1.0 - 1.0e-16]]
control_points_3 = [[0.2, 1.0 - 1.0e-16], [0.2, 1.0e-16]]
control_points_4 = [[0.2, 1.0e-16], [0.0, 0.0]]

el_size = 0.005

boundary1 = LineElement(
    control_points_1, el_size=el_size, bcond={"noslip": True, "adiabatic": True}
)
boundary2 = LineElement(
    control_points_2, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=el_size, bcond={"noslip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=el_size,
    bcond={"normal_force": surface_model, "adiabatic": True},
)
domain_boundaries = (boundary1, boundary2, boundary3, boundary4)
domain = SimpleDomain(domain_boundaries)

solver = UnsteadyTVAcousticSolver(domain, Re=5.0e3, Pr=10.0, dt=1.0e-3)
initial_state = (0.0, (0.0, 0.0), 0.0)

f = dolf.File("temp.pvd")
P = dolf.Function(solver.forms.pressure_function_space.collapse())
P.rename("P", "P")
# P.interpolate(initial_state[0])
# f << P

for i in range(1000):
    old_state = initial_state
    w = solver.solve(initial_state)

    initial_state = w.split(True)
    if i % 10 == 9:
        solver.output_field(initial_state)

    # Updating the curvature
    flow_rate = dolf.assemble(
        dolf.inner(initial_state[1], domain.n) * domain.ds((boundary4.surface_index,))
    )
    surface_model.update_curvature(flow_rate, solver._dt)

    current_time += solver._dt
    print(current_time, surface_model.kappa)
