from firecrest.fem.base_weakform import BaseWeakForm
from firecrest.fem.tv_acoustic_fspace import TVAcousticFunctionSpace
from firecrest.fem.struct_templates import AcousticConstants
import dolfin as dolf
import ufl


class TVAcousticWeakForm(BaseWeakForm):
    def __init__(self, domain, **kwargs):
        self.domain = domain
        self.function_space = TVAcousticFunctionSpace(self.domain).function_spaces
        self.trial_functions = dolf.TrialFunctions(self.function_space)
        self.test_functions = dolf.TestFunctions(self.function_space)

        self.pressure, self.velocity, self.temperature = self.trial_functions
        self.test_pressure, self.test_velocity, self.test_temperature = (
            self.test_functions
        )

        self.dolf_constants = self.get_constants(kwargs)
        self.I = dolf.Identity(self.domain.mesh.ufl_cell().geometric_dimension())

    def get_constants(self, kwargs):
        self._gamma = kwargs.get("gamma", 1.4)
        self._Re = kwargs["Re"]
        self._Pe = kwargs["Pe"]
        return AcousticConstants(
            gamma=dolf.Constant(self._gamma),
            Re=dolf.Constant(self._Re),
            Pe=dolf.Constant(self._Pe),
        )

    def density(self, pressure=None, temperature=None):
        if pressure is None and temperature is None:
            pressure, temperature = self.pressure, self.temperature

        return self.dolf_constants.gamma * pressure - temperature

    def entropy(self, pressure=None, temperature=None):
        if pressure is None and temperature is None:
            pressure, temperature = self.pressure, self.temperature

        return temperature / dolf.Constant(self.dolf_constants.gamma - 1.0) - pressure

    def shear_stress(self, velocity=None):
        if velocity is None:
            velocity = self.velocity

        i, j = ufl.indices(2)
        shear_stress = (
            velocity[i].dx(j)
            + dolf.Constant(1.0 / 3.0) * self.I[i, j] * dolf.div(velocity)
        ) / self.dolf_constants.Re
        return dolf.as_tensor(shear_stress, (i, j))

    def stress(self, pressure=None, velocity=None):
        if pressure is None and velocity is None:
            pressure, velocity = self.pressure, self.velocity

        return -pressure * self.I + self.shear_stress(velocity)

    def heat_flux(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return (
            dolf.grad(temperature)
            / self.dolf_constants.Pe
            / dolf.Constant(self.dolf_constants.gamma - 1.0)
        )

    def _unpack_functions(self, functions):
        try:
            pressure, velocity, temperature = functions
        except ValueError as v:
            print(f"Not enough values to unpack a function {functions}")
            raise v
        return (pressure, velocity, temperature)

    def spatial_component(self, trial=None, test=None):
        if trial is None:
            trial = self.trial_functions
        if test is None:
            test = self.test_functions
        pressure, velocity, temperature = trial
        test_pressure, test_velocity, test_temperature = test

        continuity_component = self.density(pressure, temperature) * test_pressure
        momementum_component = dolf.inner(velocity, test_velocity)
        energy_component = self.entropy(pressure, temperature) * test_temperature

        return continuity_component + momementum_component + energy_component

    def volume_flux_component(self, trial=None, test=None):
        if trial is None:
            trial = self.trial_functions
        if test is None:
            test = self.test_functions
        pressure, velocity, temperature = trial
        test_pressure, test_velocity, test_temperature = test

        i, j = ufl.indices(2)

        continuity_component = test_pressure * dolf.div(velocity)
        momementum_component = dolf.inner(
            dolf.as_tensor(test_velocity[i].dx(j), (i, j)),
            self.stress(pressure, velocity),
        )
        energy_component = dolf.inner(
            dolf.grad(test_temperature), self.heat_flux(temperature)
        )

        return continuity_component + momementum_component + energy_component
