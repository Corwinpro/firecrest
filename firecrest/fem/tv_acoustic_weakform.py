from firecrest.fem.base_weakform import BaseWeakForm
from firecrest.fem.tv_acoustic_fspace import TVAcousticFunctionSpace
from firecrest.fem.struct_templates import AcousticConstants
import dolfin as dolf
import ufl


class TVAcousticWeakForm(BaseWeakForm):
    def __init__(self, domain, **kwargs):
        super().__init__(domain, **kwargs)
        self.function_space_factory = TVAcousticFunctionSpace(self.domain)
        self.trial_functions = dolf.TrialFunctions(self.function_space)
        self.test_functions = dolf.TestFunctions(self.function_space)

        self.pressure, self.velocity, self.temperature = self.trial_functions
        self.test_pressure, self.test_velocity, self.test_temperature = (
            self.test_functions
        )

        self.dolf_constants = self.get_constants(kwargs)
        self.allowed_stress_bcs = {
            "noslip": "Dirichlet",
            "inflow": "Dirichlet",
            "free": "Neumann",
            "force": "Neumann",
            "impedance": "Robin",
        }

        self.allowed_temperature_bcs = {
            "isothermal": "Dirichlet",
            "temperature": "Dirichlet",
            "adiabatic": "Neumann",
            "heat_flux": "Neumann",
            "thermal_accomodation": "Robin",
        }

    @property
    def function_space(self):
        return self.function_space_factory.function_spaces

    @property
    def pressure_function_space(self):
        return self.function_space_factory.pressure_function_space

    @property
    def velocity_function_space(self):
        return self.function_space_factory.velocity_function_space

    @property
    def temperature_function_space(self):
        return self.function_space_factory.temperature_function_space

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

        return self.shear_stress(velocity) - pressure * self.I

    def heat_flux(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return (
            dolf.grad(temperature)
            / self.dolf_constants.Pe
            / dolf.Constant(self.dolf_constants.gamma - 1.0)
        )

    @staticmethod
    def _unpack_functions(functions):
        try:
            pressure, velocity, temperature = functions
        except ValueError as v:
            print(f"Not enough values to unpack a function {functions}")
            raise v
        return (pressure, velocity, temperature)

    def temporal_component(self, trial=None, test=None):
        if trial is None:
            trial = self.trial_functions
        if test is None:
            test = self.test_functions
        pressure, velocity, temperature = trial
        test_pressure, test_velocity, test_temperature = test

        continuity_component = self.density(pressure, temperature) * test_pressure
        momementum_component = dolf.inner(velocity, test_velocity)
        energy_component = self.entropy(pressure, temperature) * test_temperature

        return (
            continuity_component + momementum_component + energy_component
        ) * dolf.dx

    def spatial_component(self, trial=None, test=None):
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

        return (
            continuity_component + momementum_component + energy_component
        ) * dolf.dx

    def boundary_components(self, trial=None, test=None):
        """
        I suppose this is the place where I actually consider
        various boundary conditions and specific values for them.
        For instance, if I have a boundary element with prescribed
        velocity profile on it, I must a priory know this is a DirichletBC,
        and if I have a heat flux (Neumann) boundary condition on a wall,
        it must be hard-coded as a part of a weak form, not DirichletBC.

        I expect the usage should be something like:
            bcond = {"noslip" : True, "heat_flux" : 1.}
            bcond = {}
        """
        if trial is None:
            trial = self.trial_functions
        if test is None:
            test = self.test_functions
        _, velocity, temperature = trial
        _, test_velocity, test_temperature = test

        stress_boundary_component = dolf.Constant(0.0) * self.domain.ds
        temperature_boundary_component = dolf.Constant(0.0) * self.domain.ds
        dirichlet_bcs = []

        for boundary in self.domain.boundary_elements:
            # Step 1. Parse boundary condition data provided by boundary elements.
            # We only accept one boundary condition for stress/velocity and temperature/heat flux.
            temperature_bc = self._verify_boundary_condition(
                boundary.bcond, self.allowed_temperature_bcs
            )
            stress_bc = self._verify_boundary_condition(
                boundary.bcond, self.allowed_stress_bcs
            )
            temperature_bc_type = self.allowed_temperature_bcs[temperature_bc]
            stress_bc_type = self.allowed_stress_bcs[temperature_bc]
            # Step 2. If the boundary condition is one of the Dirichlet-compatible,
            # we construct Dirichlet boundary condition.
            if temperature_bc_type == "Dirichlet":
                if temperature_bc == "isothermal":
                    temperature = dolf.Constant(0.0)
                elif temperature_bc == "temperature":
                    temperature = dolf.Constant(boundary.bcond[temperature_bc])
                else:
                    raise TypeError(
                        f"Invalid temperature boundary condition type for {temperature_bc_type} condition."
                    )
                dirichlet_bcs.append(
                    dolf.DirichletBC(
                        self.temperature_function_space,
                        temperature,
                        self.domain.ds,
                        boundary.surface_index,
                    )
                )

            if stress_bc_type == "Dirichlet":
                if stress_bc == "noslip":
                    stress = dolf.Constant((0.0,) * self.geometric_dimension)
                elif stress_bc == "inflow":
                    stress = dolf.Constant(boundary.bcond[temperature_bc])
                else:
                    raise TypeError(
                        f"Invalid temperature boundary condition type for {stress_bc_type} condition."
                    )
                dirichlet_bcs.append(
                    dolf.DirichletBC(
                        self.velocity_function_space,
                        stress,
                        self.domain.ds,
                        boundary.surface_index,
                    )
                )

            # Step 3. If the boundary condition is one of the Neumann or Robin,
            # we construct necessary boundary integrals in weak form.
            if temperature_bc_type == "Neumann" or temperature_bc_type == "Robin":
                if temperature_bc == "adiabatic":
                    heat_flux = dolf.Constant(0.0)
                elif temperature_bc == "heat_flux":
                    heat_flux = dolf.Constant(boundary.bcond[temperature_bc])
                elif temperature_bc == "thermal_accomodation":
                    heat_flux = (
                        -dolf.Constant(boundary.bcond[temperature_bc]) * temperature
                    )
                else:
                    raise TypeError(
                        f"Invalid temperature boundary condition type for {temperature_bc_type} condition."
                    )
                temperature_boundary_component += (
                    -heat_flux
                    * test_temperature
                    * self.domain.ds((boundary.surface_index,))
                )

            if stress_bc_type == "Neumann" or stress_bc_type == "Robin":
                if stress_bc == "free":
                    stress = dolf.Constant((0.0,) * self.geometric_dimension)
                elif stress_bc == "force":
                    stress = dolf.Constant(boundary.bcond[stress_bc])
                elif stress_bc == "impedance":
                    stress = dolf.Constant(boundary.bcond[stress_bc]) * velocity
                else:
                    raise TypeError(
                        f"Invalid temperature boundary condition type for {stress_bc_type} condition."
                    )
                stress_boundary_component += -dolf.inner(
                    stress, test_velocity
                ) * self.domain.ds((boundary.surface_index,))

        return dirichlet_bcs, stress_boundary_component, temperature_boundary_component

    @staticmethod
    def _verify_boundary_condition(bcond, allowed_bconds):
        bc = allowed_bconds.intersection(bcond)
        if len(bc) != 1:
            raise TypeError(
                "Incorrect number of temperature-related boundary condition."
                f"One expected, {len(bc)} received."
            )
        return bc
