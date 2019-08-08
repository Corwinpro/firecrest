import math


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

    def volume_derivative(self, kappa=None):
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
            + self.constants.acoustic_mach
            * dt
            * flow_rate
            / self.volume_derivative(self.kappa)
        )
        self.kappa_history.append(kappa_updated)
