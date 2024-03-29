from math import pi as PI
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


class SurfaceModelFactory:
    def __init__(self, model_data, constants_data):
        self.initial_condition = model_data["initial_curvature"]
        nozzle_domain_length = model_data["length"]
        nozzle_domain_radius = model_data["radius"]

        L = constants_data["length"]
        c_s = constants_data["sound_speed"]
        rho = constants_data["density"]
        epsilon = constants_data["Mach"]
        gamma_st = constants_data["surface_tension"]
        mu = constants_data["viscosity"]
        Re = rho * c_s * L / mu

        self.constants = Constants(
            rho / rho,
            epsilon,
            L / L,
            nozzle_domain_length / L,
            nozzle_domain_radius / L,
            2.0 * gamma_st / (rho * c_s ** 2.0 * nozzle_domain_radius * epsilon),
            Re,
            c_s / c_s,
        )

    def create_direct_model(self):
        return SurfaceModel(constants=self.constants, kappa_t0=self.initial_condition)

    def create_adjoint_model(self, direct_surface):
        return AdjointSurfaceModel(direct_surface=direct_surface)


class SurfaceModel:
    def __init__(self, constants, kappa_t0=None, kappa_history=None):
        """
        Params:
            - kappa_t0: initial curvature of the free surface, kappa(t = t_0)
            - kappa_history: if we create a surface with a known history

        Attributes:
            - default_R: non-dimensional radius of the nozzle or the radius of the unperturbed (flat) free surface
            - gamma_tension: non-dimensional surface tension coefficient
        """
        self.constants = constants
        self.r = self.constants.nozzle_R
        self.epsilon = self.constants.acoustic_mach
        self.gamma_tension = self.constants.surface_tension
        if kappa_history is not None:
            self.kappa_history = kappa_history
        else:
            if kappa_t0 is None:
                self.kappa_t0 = 0.05  # Just an arbitrary non zero surface curvature
            else:
                self.kappa_t0 = kappa_t0
            self.kappa_history = [self.kappa_t0]

    def eval(self):
        return -self.pressure

    @property
    def prev_kappa(self):
        try:
            return self.kappa_history[-2]
        except IndexError:
            return self.kappa

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
        return 0.5 * (self.kappa + self.prev_kappa) * self.gamma_tension

    @staticmethod
    def _cos_theta(kappa):
        return (1.0 - kappa ** 2.0) ** 0.5

    def volume_derivative(self, kappa=None):
        """
        This implements a 3D spherical nozzle cap shape model.
        Calculate the derivative of the volume of the nozzle with respect to the free surface curvature.
        """
        if kappa is None:
            kappa = self.kappa

        if abs(kappa) < 1.0e-3:
            return (
                2.0
                * PI
                * self.r ** 3.0
                / 8
                * (
                    1.0
                    + kappa ** 2.0
                    + 15.0 / 16.0 * kappa ** 4
                    + 7.0 / 8.0 * kappa ** 6.0
                )
            )
        else:
            return (
                8.0
                * PI
                / kappa ** 4.0
                * (1.0 - self._cos_theta(kappa)) ** 2.0
                / self._cos_theta(kappa)
                * self.r ** 3.0
                / 8
            )

    def second_volume_derivative(self, kappa):
        return (
            8.0 * PI * self.r ** 3.0 / 8.0
            * (1.0 - self._cos_theta(kappa))
            / (kappa ** 3.0 * self._cos_theta(kappa))
            * (
                (1.0 - self._cos_theta(kappa)) / self._cos_theta(kappa) ** 2.0
                - 4.0 * (1.0 - self._cos_theta(kappa)) / kappa ** 2.0
                + 2.0 / self._cos_theta(kappa)
            )
        )

    def surface_energy(self, kappa=None):
        """
        Calculates the surface energy of the nozzle flow, minus the energy of the flat surface
        of the flat surface
        """
        if kappa is None:
            kappa = 0.5 * (self.kappa + self.prev_kappa)

        if abs(kappa) < 1.0e-3:
            _kappa_function = (
                0.5
                + kappa ** 2.0 / 8.0
                + kappa ** 4.0 / 16.0
                + kappa ** 6.0 * 5.0 / 128.0
            )
        else:
            _kappa_function = (1.0 - self._cos_theta(kappa)) / kappa ** 2.0

        energy = (
            self.r ** 3.0
            / (8.0 * self.epsilon)
            * (self.gamma_tension * 8.0 * PI * _kappa_function)
        )
        static_energy = (
            self.r ** 3.0 / (8.0 * self.epsilon) * self.gamma_tension * 4.0 * PI
        )

        return energy - static_energy

    def update_curvature(self, flow_rate, dt):
        """
        Given a velocity at the control boundary, update the curvature (== volume, or mass) inside the nozzle
        """
        kappa_updated = (
            self.kappa
            + self.constants.acoustic_mach
            * dt
            * flow_rate
            / self.volume_derivative(self.kappa)
        )
        # Second order correction
        kappa_updated += (
            -dt ** 2.0 / 2.0
            * ((self.kappa - self.prev_kappa)/dt) ** 2.0
            * self.second_volume_derivative(self.kappa)
            / self.volume_derivative(self.kappa)
        )
        self.kappa_history.append(kappa_updated)


class AdjointSurfaceModel:
    """
    The AdjointSurfaceModel is the adjoint counterpart of the direct surface model,
    and uses the values of the direct curvature at each time step.
    """

    def __init__(self, direct_surface: SurfaceModel):
        self.direct_surface = direct_surface
        self.kappa_adj = self.direct_surface.kappa_history[-1]
        self.kappa_adj_history = []

    def eval(self):
        return -self.adj_pressure

    @property
    def adj_pressure(self):
        """
        Adjoint pressure created by the surface. We use it as the adjoint boundary condition.
        """
        if len(self.kappa_adj_history) < 2:
            return self.kappa_adj * self.direct_surface.gamma_tension
        return (
            0.5
            * (self.kappa_adj + self.kappa_adj_history[-2])
            * self.direct_surface.gamma_tension
        )

    def update_curvature(self, adjoint_flow_rate, dt):
        """
        Updates the adjoint curvature, which 'follows' the trajectory of the direct (real) curvature
        """
        kappa_n = self.direct_surface.kappa_history[-len(self.kappa_adj_history) - 1]
        try:
            kappa_prev = self.direct_surface.kappa_history[
                -len(self.kappa_adj_history) - 2
            ]
        except IndexError:
            return

        self.kappa_adj = self.kappa_adj * self.direct_surface.volume_derivative(
            kappa=kappa_n
        ) / self.direct_surface.volume_derivative(kappa=kappa_prev) + (
            self.direct_surface.constants.acoustic_mach
            * dt
            * adjoint_flow_rate
            / self.direct_surface.volume_derivative(kappa=kappa_prev)
        )
        self.kappa_adj_history.append(self.kappa_adj)
