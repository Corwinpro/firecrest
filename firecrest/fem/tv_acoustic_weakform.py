import functools
from abc import ABC, abstractmethod
from firecrest.fem.base_weakform import BaseWeakForm
from firecrest.fem.tv_acoustic_fspace import (
    TVAcousticFunctionSpace,
    ComplexTVAcousticFunctionSpace,
)
from firecrest.misc.matrix_manipulations import (
    vector_to_ndarray,
    boundary_dofs,
    outer_to_matrix,
)
from firecrest.fem.struct_templates import AcousticConstants
import dolfin as dolf
import ufl

#: Default value of gamma = cp/cv for water
DEFAULT_CP_CV_RATIO = 1.017


def parse_trialtest(component):
    """
    Parse Real / Complex case of trial and test functions arguments for weak forms.
    """

    @functools.wraps(component)
    def inner(self, trial=None, test=None, *args, **kwargs):
        if trial is None and test is None:
            """
            With empty arguments we implement default trial/test complex component
            """
            trial = self.trial_functions
            test = self.test_functions
        elif test is None:
            """
            The test is not specified when we need to calculate linear forms with
            existing test functions (e.g., linear form from previous time step in unsteady problem). 
            """
            test = self.test_functions[: len(trial)]

        if len(trial) == 3 and len(test) == 3:
            """
            If triplets trial and test are given, we construct real component
            """
            return component(self, trial, test)
        elif len(trial) == 6 and len(test) == 6:
            """
            If triplets*2 trial and test are given, we construct complex component
            """
            _flag = self.complex_forms_flag
            self.complex_forms_flag = "real"
            forms = component(self, trial[:3], test[:3], *args, **kwargs) + component(
                self, trial[3:], test[3:], *args, **kwargs
            )
            self.complex_forms_flag = "imag"
            forms += component(self, trial[:3], test[3:], *args, **kwargs) - component(
                self, trial[3:], test[:3], *args, **kwargs
            )
            self.complex_forms_flag = _flag
            return forms
        elif (trial in ["real", "imag"]) and (test in ["real", "imag"]):
            return component(
                self,
                self.complex_component(self.trial_functions, trial),
                self.complex_component(self.test_functions, test),
                *args,
                **kwargs,
            )
        else:
            raise ValueError

    return inner


class BaseTVAcousticWeakForm(ABC, BaseWeakForm):
    """
    Base class for Thermoviscous acoustic weak forms generation.
    """

    allowed_stress_bcs = {
        "noslip": "Dirichlet",
        "inflow": "Dirichlet",
        "free": "Neumann",
        "force": "Neumann",
        "normal_force": "Neumann",
        "impedance": "Robin",
        "inhom_impedance": "Robin",
        "nozzle_impedance": "Robin",
        "inhom_nozzle_impedance": "Robin",
        "normal_impedance": "Robin",
        "shape_impedance": "Robin",
        "slip": "Neumann",
        "normal_velocity": "Neumann",
        "avg_velocity": None,
    }

    allowed_temperature_bcs = {
        "isothermal": "Dirichlet",
        "temperature": "Dirichlet",
        "adiabatic": "Neumann",
        "heat_flux": "Neumann",
        "thermal_accommodation": "Robin",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dolf_constants = self.get_constants(kwargs)
        self.trial_functions = dolf.TrialFunctions(self.function_space)
        self.test_functions = dolf.TestFunctions(self.function_space)

    def get_constants(self, kwargs):
        self._gamma = kwargs.get("gamma", DEFAULT_CP_CV_RATIO)
        self._Re = kwargs["Re"]
        try:
            self._Pe = kwargs["Pe"]
        except KeyError:
            try:
                _Pr = kwargs["Pr"]
                self._Pe = self._Re * _Pr
            except KeyError:
                raise KeyError(
                    "Peclet or Prandtl numbers required for Thermoviscous acoustic flow."
                )
        return AcousticConstants(
            gamma=dolf.Constant(self._gamma),
            Re=dolf.Constant(self._Re),
            Pe=dolf.Constant(self._Pe),
        )

    @property
    def function_space(self):
        try:
            return self.function_space_factory.function_spaces
        except AttributeError:
            raise AttributeError(
                "A function_space_factory which implements "
                "`functions_spaces` must be provided."
            )

    @property
    @abstractmethod
    def function_space_factory(self):
        pass

    @property
    def pressure_function_space(self):
        return self.function_space_factory.pressure_function_space

    @property
    def velocity_function_space(self):
        return self.function_space_factory.velocity_function_space

    @property
    def temperature_function_space(self):
        return self.function_space_factory.temperature_function_space

    def density(self, pressure, temperature):
        return self.dolf_constants.gamma * pressure - temperature

    def entropy(self, pressure, temperature):
        return temperature / dolf.Constant(self.dolf_constants.gamma - 1.0) - pressure

    def shear_stress(self, velocity):
        if self.geometric_dimension == 1:
            return dolf.Constant(4.0 / 3.0) * velocity.dx(0) / self.dolf_constants.Re

        i, j = ufl.indices(2)
        shear_stress = (
            velocity[i].dx(j)
            + dolf.Constant(1.0 / 3.0) * self.identity_tensor[i, j] * dolf.div(velocity)
        ) / self.dolf_constants.Re
        # shear_stress = (
        #     velocity[i].dx(j)
        #     + velocity[j].dx(i)
        #     - dolf.Constant(2.0 / 3.0) * self.identity_tensor[i, j] * dolf.div(velocity)
        # ) / self.dolf_constants.Re
        return dolf.as_tensor(shear_stress, (i, j))

    def stress(self, pressure, velocity):
        if self.geometric_dimension == 1:
            return self.shear_stress(velocity) - pressure
        return self.shear_stress(velocity) - pressure * self.identity_tensor

    def kinetic_energy_flux(self, state, direction, boundary):
        """
        Calculates the kinetic energy flux u_i * sigma_ij * n_j through the
        `boundary`
        :param state: self state
        :param direction: flux direction n_j
        :param boundary: target surface
        :return: float energy flux
        """
        pressure, velocity, temperature = state
        flux = dolf.inner(
            velocity,
            dolf.dot(dolf.as_vector(direction), self.stress(pressure, velocity)),
        )
        return dolf.assemble(flux * self.domain.ds((boundary.surface_index,)))

    def mass_flow_rate(self, state, boundary):
        return dolf.assemble(
            dolf.inner(state[1], self.domain.n)
            * self.domain.ds((boundary.surface_index,))
        )

    def avg_normal_stress(self, state, boundary):
        stress = self.stress(state[0], state[1])
        stress = dolf.assemble(
            dolf.dot(dolf.dot(stress, self.domain.n), self.domain.n)
            * self.domain.ds((boundary.surface_index,))
        )
        return stress

    def weighted_stress(self, state, boundary, weight_vector_function):
        stress = self.stress(state[0], state[1])
        stress = dolf.assemble(
            dolf.dot(dolf.dot(stress, self.domain.n), weight_vector_function)
            * self.domain.ds((boundary.surface_index,))
        )
        return stress

    def heat_flux(self, temperature):
        return (
            dolf.grad(temperature)
            / self.dolf_constants.Pe
            / dolf.Constant(self.dolf_constants.gamma - 1.0)
        )

    def _temporal_component(self, trial, test):
        """
        Generate the forms for the temporal component of the TVAcoustic weak form equation.
        """
        pressure, velocity, temperature = trial
        test_pressure, test_velocity, test_temperature = test

        continuity_component = self.density(pressure, temperature) * test_pressure
        momentum_component = dolf.inner(velocity, test_velocity)
        energy_component = self.entropy(pressure, temperature) * test_temperature
        return continuity_component + momentum_component + energy_component

    def _spatial_component(self, trial, test):
        pressure, velocity, temperature = trial
        test_pressure, test_velocity, test_temperature = test

        i, j = ufl.indices(2)

        if self.geometric_dimension == 1:
            continuity_component = test_pressure * velocity.dx(0)
            momentum_component = test_velocity.dx(0) * self.stress(pressure, velocity)
        else:
            continuity_component = test_pressure * dolf.div(velocity)
            momentum_component = dolf.inner(
                dolf.as_tensor(test_velocity[i].dx(j), (i, j)),
                self.stress(pressure, velocity),
            )
        energy_component = dolf.inner(
            dolf.grad(test_temperature), self.heat_flux(temperature)
        )
        return continuity_component + momentum_component + energy_component

    def temporal_component(self, trial, test, shift=None):
        """
        Generate the volume-averaged forms for the temporal component of the
        TVAcoustic weak form equation.
        """
        components = self._temporal_component(trial, test)

        if shift is None:
            shift = 1.0

        return self._parse_dolf_expression(shift) * components * self.domain.dx

    def spatial_component(self, trial, test, shift=None):
        """
        Generate volume-averaged spatial component of the TVAcoustic weak form equation.
        """
        components = self._spatial_component(trial, test)

        if shift is None:
            shift = 1.0

        return self._parse_dolf_expression(shift) * components * self.domain.dx

    def volume_dissipation(self, state):
        return dolf.assemble(self.spatial_component(state, state))

    def boundary_components(self, trial, test):
        """
        Generates momentum (stress, velocity) and thermal (heat flux. temperature) boundary components
        of the TVAcoustic weak form equation.

        I expect the usage should be something like:
            bcond = {"noslip" : True, "heat_flux" : 1.}
        """
        pressure, velocity, temperature = trial
        test_pressure, test_velocity, test_temperature = test

        stress_boundary_component = dolf.Constant(0.0) * self.domain.ds()
        temperature_boundary_component = dolf.Constant(0.0) * self.domain.ds()

        for boundary in self.domain.boundary_elements:
            # Step 1. Parse boundary condition data provided by boundary elements.
            # We only accept one boundary condition for stress/velocity and temperature/heat flux.
            temperature_bc = self._pop_boundary_condition(
                boundary.bcond, self.allowed_temperature_bcs
            )
            temperature_bc_type = self.allowed_temperature_bcs[temperature_bc]

            # Step 2. If the boundary condition is one of the Neumann or Robin,
            # we construct necessary boundary integrals in weak form.
            if temperature_bc_type == "Neumann" or temperature_bc_type == "Robin":
                if temperature_bc == "adiabatic":
                    heat_flux = dolf.Constant(0.0)
                elif temperature_bc == "heat_flux":
                    heat_flux = self._parse_dolf_expression(
                        boundary.bcond[temperature_bc]
                    )
                elif temperature_bc == "thermal_accommodation":
                    heat_flux = (
                        -self._parse_dolf_expression(boundary.bcond[temperature_bc])
                        * temperature
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

            # Same for stress / velocity boundary conditions
            stress_bc = self._pop_boundary_condition(
                boundary.bcond, self.allowed_stress_bcs
            )
            stress_bc_type = self.allowed_stress_bcs[stress_bc]

            if stress_bc_type == "Neumann" or stress_bc_type == "Robin":
                if stress_bc == "free":
                    stress = dolf.Constant((0.0,) * self.geometric_dimension)
                elif stress_bc == "force":
                    stress = self._parse_dolf_expression(boundary.bcond[stress_bc])
                elif stress_bc == "impedance":
                    stress = (
                        self._parse_dolf_expression(boundary.bcond[stress_bc])
                        * velocity
                    )
                elif stress_bc == "inhom_impedance":
                    # Impedance term
                    stress = (
                        self._parse_dolf_expression(boundary.bcond[stress_bc][0])
                        * velocity
                    )
                    # Force term
                    normal_force = boundary.bcond[stress_bc][1]
                    normal_force = 0.5 * (
                        normal_force.real + normal_force.imag
                    ) - 0.5j * (normal_force.real - normal_force.imag)
                    stress += self._parse_dolf_expression(normal_force) * self.domain.n
                elif stress_bc == "nozzle_impedance":
                    capacitance = boundary.bcond[stress_bc]["capacitance"]
                    inductance = boundary.bcond[stress_bc]["inductance"]
                    resistance = boundary.bcond[stress_bc]["resistance"]
                    frequency = boundary.bcond[stress_bc]["frequency"]

                    z1 = -1.0 / (frequency * capacitance) - frequency * inductance
                    z1 = self._parse_dolf_expression(z1)

                    z2 = self._parse_dolf_expression(resistance)

                    stress = z1 * dolf.inner(
                        velocity, self.domain.n
                    ) * self.domain.n + z2 * velocity.dx(0).dx(0)
                elif stress_bc == "inhom_nozzle_impedance":
                    inductance = boundary.bcond[stress_bc]["inductance"]
                    resistance = boundary.bcond[stress_bc]["resistance"]
                    frequency = boundary.bcond[stress_bc]["frequency"]

                    z1 = -1.0j * frequency * inductance
                    z1 = self._parse_dolf_expression(z1)

                    z2 = self._parse_dolf_expression(resistance)

                    stress = z1 * velocity + z2 * velocity.dx(0).dx(0)

                    # Force term
                    normal_force = boundary.bcond[stress_bc]["force"]
                    normal_force = 0.5 * (
                        normal_force.real + normal_force.imag
                    ) - 0.5j * (normal_force.real - normal_force.imag)
                    stress += self._parse_dolf_expression(normal_force) * self.domain.n
                elif stress_bc == "shape_impedance":
                    try:
                        shape = boundary.velocity_shape
                    except AttributeError:
                        raise AttributeError(
                            "`shape_impedance` boundary condition requires expected velocity shape. "
                            + "Provide it as a dolfin Expression."
                        )
                    stress = (
                        self._parse_dolf_expression(boundary.bcond[stress_bc])
                        * velocity
                        / shape
                    )
                elif stress_bc == "normal_impedance":
                    stress = (
                        self._parse_dolf_expression(boundary.bcond[stress_bc])
                        * dolf.inner(velocity, self.domain.n)
                        * self.domain.n
                    )
                elif stress_bc == "normal_force":
                    normal_force = boundary.bcond[stress_bc]
                    # normal_force = 0.5 * (
                    #     normal_force.real + normal_force.imag
                    # ) - 0.5j * (normal_force.real - normal_force.imag)
                    stress = self._parse_dolf_expression(normal_force) * self.domain.n
                elif stress_bc == "slip" or stress_bc == "normal_velocity":
                    normal_velocity = (
                        dolf.Constant(0.0)
                        if stress_bc == "slip"
                        else self._parse_dolf_expression(boundary.bcond[stress_bc])
                    )
                    stress = (
                        dolf.Constant(-1.0e2)
                        / dolf.CellDiameter(self.domain.mesh)
                        * (dolf.inner(velocity, self.domain.n) - normal_velocity)
                        * self.domain.n
                    )
                else:
                    raise TypeError(
                        f"Invalid temperature boundary condition type for {stress_bc_type} condition."
                    )
                stress_boundary_component += -dolf.inner(
                    stress, test_velocity
                ) * self.domain.ds((boundary.surface_index,))

        return stress_boundary_component + temperature_boundary_component

    @staticmethod
    def _pop_boundary_condition(bcond, allowed_bconds):
        """
        Verification of a proper number of boundary conditions of one type.
        If we are not given a single boundary condition of a single type, something is wrong. 
        """
        bc = set(bcond.keys()) & set(allowed_bconds.keys())
        if len(bc) != 1:
            raise TypeError(
                "Incorrect number of boundary conditions."
                f"Expected 1, but {len(bc)} received."
            )
        return bc.pop()

    @property
    def bc_to_fs(self):
        """
        Maps the name of the boundary condition to the corresponding function space
        :return: the mapping dictionary
        """
        return {
            "noslip": self.velocity_function_space,
            "inflow": self.velocity_function_space,
            "isothermal": self.temperature_function_space,
            "temperature": self.temperature_function_space,
        }

    def dirichlet_boundary_conditions(self, is_linearised=False):
        """
        Generates DirichletBCs on all appropriate boundary elements.
        :param is_linearised: if True, the homogeneous boundary conditions will be generated by default
        """
        dirichlet_bcs = []
        # Parse boundary condition data provided by boundary elements.
        for boundary in self.domain.boundary_elements:
            # We only accept one Dirichlet boundary condition for temperature.
            temperature_bc = self._pop_boundary_condition(
                boundary.bcond, self.allowed_temperature_bcs
            )

            temperature_bc_type = self.allowed_temperature_bcs[temperature_bc]
            if temperature_bc_type == "Dirichlet":
                dirichlet_bcs.extend(
                    self._generate_dirichlet_bc(boundary, temperature_bc, is_linearised)
                )
            # We only accept one Dirichlet boundary condition for velocity.
            velocity_bc = self._pop_boundary_condition(
                boundary.bcond, self.allowed_stress_bcs
            )

            velocity_bc_type = self.allowed_stress_bcs[velocity_bc]
            if velocity_bc_type == "Dirichlet":
                dirichlet_bcs.extend(
                    self._generate_dirichlet_bc(boundary, velocity_bc, is_linearised)
                )

        return dirichlet_bcs

    def _generate_dirichlet_bc(self, boundary, bc_type, is_linearised=False):
        """
        Given a boundary and a boundary condition type (one from the 'bc_to_fs' dict),
        we generate a dolfin DirichletBC based on the boundary expression for this boundary condition.
        """

        if bc_type == "noslip" or (bc_type == "inflow" and is_linearised):
            if self.geometric_dimension == 1:
                value = dolf.Constant(0.0)
            else:
                value = dolf.Constant((0.0,) * self.geometric_dimension)
        elif bc_type == "isothermal" or (bc_type == "temperature" and is_linearised):
            value = dolf.Constant(0.0)
        elif bc_type == "inflow" or bc_type == "temperature":
            value = self._parse_dolf_expression(boundary.bcond[bc_type])
        else:
            raise TypeError(f"Invalid boundary condition type for Dirichlet condition.")

        function_space = self.bc_to_fs[bc_type]
        return [
            dolf.DirichletBC(
                function_space,
                value,
                self.domain.boundary_parts,
                boundary.surface_index,
            )
        ]

    def _lhs_forms(self):
        """
        Constructs the LHS forms (spatial components), AA of the problem AA*x = (...)*BB*x.
        """
        return self.spatial_component() + self.boundary_components()

    def _rhs_forms(self, shift=None):
        """
        Constructs the RHS forms (temporal components), BB of the problem AA*x = (...)*BB*x.
        """
        return self.temporal_component(shift=shift)


class TVAcousticWeakForm(BaseTVAcousticWeakForm):
    function_space_factory = None

    def __init__(self, domain, **kwargs):
        self.function_space_factory = TVAcousticFunctionSpace(domain)
        super().__init__(domain, **kwargs)

    @staticmethod
    def _unpack_functions(functions):
        try:
            pressure, velocity, temperature = functions
        except ValueError as v:
            print(f"Not enough values to unpack a function {functions}")
            raise v
        return pressure, velocity, temperature

    @parse_trialtest
    def temporal_component(self, trial=None, test=None):
        return super().temporal_component(trial, test)

    @parse_trialtest
    def spatial_component(self, trial=None, test=None):
        return super().spatial_component(trial, test)

    @parse_trialtest
    def boundary_components(self, trial=None, test=None):
        return super().boundary_components(trial, test)

    def energy(self, state):
        return dolf.assemble(self.temporal_component(state, state)) / 2.0


class ComplexTVAcousticWeakForm(BaseTVAcousticWeakForm):
    """
    Weak forms for complex TVAcoustic problem.
    """

    function_space_factory = None

    def __init__(self, domain, **kwargs):
        self.function_space_factory = ComplexTVAcousticFunctionSpace(domain)
        super().__init__(domain, **kwargs)
        self.real_function_space_factory = None

    @staticmethod
    def complex_component(functions, complex_flag):
        if complex_flag == "real":
            return functions[:3]
        elif complex_flag == "imag":
            return functions[3:6]
        else:
            raise ValueError("Only 'real' or 'imag' complex flags are accepted.")

    @property
    def real_function_space(self):
        if self.real_function_space_factory is None:
            self.real_function_space_factory = TVAcousticFunctionSpace(self.domain)

        return self.real_function_space_factory.function_spaces

    @parse_trialtest
    def temporal_component(self, trial=None, test=None, shift=None):
        return super().temporal_component(trial, test, shift=shift)

    @parse_trialtest
    def spatial_component(self, trial=None, test=None):
        return super().spatial_component(trial, test)

    @parse_trialtest
    def boundary_components(self, trial=None, test=None):
        return super().boundary_components(trial, test)

    def boundary_averaged_velocity(self, trial=None, test=None):
        if trial is None:
            trial = self.trial_functions
        if test is None:
            test = self.test_functions
        u_r, u_i = trial[1], trial[4]
        v_r, v_i = test[1], test[4]

        target_boundary = None
        for boundary in self.domain.boundary_elements:
            stress_bc = self._pop_boundary_condition(
                boundary.bcond, self.allowed_stress_bcs
            )
            if stress_bc == "avg_velocity":
                target_boundary = boundary
                break

        if target_boundary is None:
            return None

        _flag = self.complex_forms_flag
        self.complex_forms_flag = "real"

        dofs_r = boundary_dofs(
            self.domain.boundary_parts,
            target_boundary,
            self.velocity_function_space,
            (1, 1),
        )

        alpha_r = self._parse_dolf_expression(target_boundary.bcond[stress_bc])

        self.complex_forms_flag = "imag"

        dofs_i = boundary_dofs(
            self.domain.boundary_parts,
            target_boundary,
            self.velocity_function_space,
            (1, 1),
        )

        alpha_i = self._parse_dolf_expression(target_boundary.bcond[stress_bc])

        self.complex_forms_flag = _flag

        averaged_velocity_r_r = dolf.assemble(
            alpha_r
            * dolf.inner(u_r, self.domain.n)
            * self.domain.ds((target_boundary.surface_index))
        )
        averaged_velocity_r_i = dolf.assemble(
            alpha_i
            * dolf.inner(u_r, self.domain.n)
            * self.domain.ds((target_boundary.surface_index))
        )
        averaged_velocity_i_r = dolf.assemble(
            alpha_r
            * dolf.inner(u_i, self.domain.n)
            * self.domain.ds((target_boundary.surface_index))
        )
        averaged_velocity_i_i = dolf.assemble(
            alpha_i
            * dolf.inner(u_i, self.domain.n)
            * self.domain.ds((target_boundary.surface_index))
        )
        averaged_test_velocity_r = dolf.assemble(
            dolf.inner(v_r, self.domain.n)
            * self.domain.ds((target_boundary.surface_index))
        )
        averaged_test_velocity_i = dolf.assemble(
            dolf.inner(v_i, self.domain.n)
            * self.domain.ds((target_boundary.surface_index))
        )
        averaged_u_r_r = vector_to_ndarray(averaged_velocity_r_r, dofs_r)
        averaged_u_r_i = vector_to_ndarray(averaged_velocity_r_i, dofs_r)
        averaged_u_i_r = vector_to_ndarray(averaged_velocity_i_r, dofs_i)
        averaged_u_i_i = vector_to_ndarray(averaged_velocity_i_i, dofs_i)

        averaged_v_r = vector_to_ndarray(averaged_test_velocity_r, dofs_r)
        averaged_v_i = vector_to_ndarray(averaged_test_velocity_i, dofs_i)

        averaged_matrix = outer_to_matrix(
            self.function_space, averaged_u_r_r, averaged_v_r
        )

        averaged_matrix.axpy(
            -1.0, outer_to_matrix(self.function_space, averaged_u_i_i, averaged_v_r)
        )
        averaged_matrix.axpy(
            1.0, outer_to_matrix(self.function_space, averaged_u_r_i, averaged_v_i)
        )
        averaged_matrix.axpy(
            1.0, outer_to_matrix(self.function_space, averaged_u_i_r, averaged_v_i)
        )

        # averaged_matrix.axpy(1.0, penalty_matrix)

        # from petsc4py import PETSc
        # myviewer = PETSc.Viewer().createASCII("test2.txt")
        # averaged_matrix.view(myviewer)
        return averaged_matrix

    @property
    def velocity_function_space(self):
        complex_fspace = super().velocity_function_space
        if self.complex_forms_flag == "real":
            return complex_fspace[0]
        elif self.complex_forms_flag == "imag":
            return complex_fspace[1]

    @property
    def temperature_function_space(self):
        complex_fspace = super().temperature_function_space
        if self.complex_forms_flag == "real":
            return complex_fspace[0]
        elif self.complex_forms_flag == "imag":
            return complex_fspace[1]

    @property
    def pressure_function_space(self):
        complex_fspace = super().pressure_function_space
        if self.complex_forms_flag == "real":
            return complex_fspace[0]
        elif self.complex_forms_flag == "imag":
            return complex_fspace[1]

    def _generate_dirichlet_bc(self, boundary, bc_type, is_linearised=False):
        """
        Generates a complex boundary condition with the use of the `complex_forms_flag`.
        """
        _flag = self.complex_forms_flag
        self.complex_forms_flag = "real"
        bc = super()._generate_dirichlet_bc(
            boundary, bc_type, is_linearised=is_linearised
        )

        self.complex_forms_flag = "imag"
        bc += super()._generate_dirichlet_bc(
            boundary, bc_type, is_linearised=is_linearised
        )

        self.complex_forms_flag = _flag
        return bc
