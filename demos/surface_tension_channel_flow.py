from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
import dolfin as dolf
import math
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


class SurfaceModel:
    def __init__(self, kappa_t0=None, kappa_history=None):
        """
        Params:
            - kappa_t0: initial curvature of the free surface, kappa(t = t_0)
            - kappa_history: if we create a surface with a known history

        Attributes:
            - default_R: nondimensional radius of the nozzle or the radius of the unperturbed (flat) free surface
            - gamma_tension: non-dimensional surface tension coefficient
        """
        self.r = nondim_constants.nozzle_R
        self.epsilon = nondim_constants.acoustic_mach
        self.gamma_tension = nondim_constants.surface_tension
        if kappa_history is not None:
            self.kappa_history = kappa_history
        else:
            if kappa_t0 is None:
                self.kappa_t0 = 0.05  # Just a random non zero surface curvature
            else:
                self.kappa_t0 = kappa_t0
            self.kappa_history = [self.kappa_t0]

    def eval(self):
        return -self.pressure

    @property
    def kappa(self):
        """
        Current (latest) curvature of the free surface, kappa(t = t_now)
        """
        return self.kappa_history[-1]

    @property
    def pressure(self):
        """
        Value of the physical pressure created by the surface
        """
        return self.kappa * self.gamma_tension

    @staticmethod
    def _cos_theta(kappa):
        return (1.0 - kappa ** 2.0) ** 0.5

    def dOmega_dkappa(self, kappa=None):
        """
        This implements a 3D spherical nozzle cap shape model.
        Calculate the derivative of the volume of the nozzle with respect to the free surface curvature.
        """
        if kappa is None:
            kappa = self.kappa

        if kappa < 1.0e-6:
            return 2.0 * math.pi * self.r ** 3.0 / 8
        else:
            return (
                8.0
                * math.pi
                / kappa ** 4.0
                * (1.0 - self._cos_theta(kappa)) ** 2.0
                / self._cos_theta(kappa)
                * self.r ** 3.0
                / 8
            )

    def surface_energy(self, kappa):
        """
        Calculates the surface energy of the nozzle flow, minus the energy of the flat surface
        of the flat surface
        """
        if kappa is None:
            kappa = self.kappa

        energy = (
            self.r ** 3.0
            / (8.0 * self.epsilon)
            * (
                self.gamma_tension
                * 8.0
                * math.pi
                / kappa ** 2.0
                * (1.0 - self._cos_theta(kappa))
            )
        )
        static_energy = (
            self.r ** 3.0 / (8.0 * self.epsilon) * self.gamma_tension * 4.0 * math.pi
        )

        return energy - static_energy

    def update_curvature(self, flow_rate, dt):
        """
        Given a velocity at the control boundary, update the curvature (== volume, or mass) inside the nozzle
        """
        kappa_updated = (
            self.kappa
            + nondim_constants.acoustic_mach
            * dt
            * flow_rate
            / self.dOmega_dkappa(self.kappa)
        )
        self.kappa_history.append(kappa_updated)


surface_model = SurfaceModel()

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

solver = UnsteadyTVAcousticSolver(domain, Re=5.0e3, Pr=10., dt=1.0e-3)
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
