from .base_assembler import BaseAssembler


class SymmetricInjectorAssembler(BaseAssembler):
    def generate_control_points(self):
        channel_height = self.geometry_data["channel_height"]
        width = self.geometry_data["channel_width"]
        actuator_length = self.geometry_data["actuator_length"]
        actuator_slip = self.geometry_data.get("actuator_slip", 0.0)
        nozzle_r = self.geometry_data["nozzle_r"]
        nozzle_l = self.geometry_data["nozzle_l"]
        nozzle_offset = self.geometry_data["nozzle_offset"]

        cp_noslip_wall = [
            [width - actuator_length - actuator_slip, channel_height + 1.0e-16],
            [0.0, channel_height],
            [0.0, 0.0],
            [width - nozzle_r - nozzle_offset, 0.0],
        ]
        cp_nozzle_noslip_wall = [
            [width - nozzle_r - nozzle_offset, 0.0],
            [width - nozzle_r, 0.0],
            [width - nozzle_r, -nozzle_l],
        ]
        cp_nozzle_free_surface = [
            [width - nozzle_r, -nozzle_l],
            [width, -nozzle_l + 1.0e-16],
        ]
        cp_symmetry_plane = [
            [width, -nozzle_l + 1.0e-16],
            [width, channel_height + 1.0e-16],
        ]
        cp_actuator = [
            [width, channel_height + 1.0e-16],
            [width - actuator_length, channel_height],
        ]

        control_points = []

        if actuator_slip > 0.0:
            cp_slip_wall = [
                [width - actuator_length, channel_height],
                [width - actuator_length - actuator_slip, channel_height + 1.0e-16],
            ]
            control_points.append(cp_slip_wall)

        control_points.extend(
            [
                cp_noslip_wall,
                cp_nozzle_noslip_wall,
                cp_nozzle_free_surface,
                cp_symmetry_plane,
                cp_actuator,
            ]
        )

        return control_points

    def generate_boundary_metadata(self):
        outer_slip_data = self._generate_boundary_dict(
            name="outer_slip",
            default_element_size=self.element_size,
            default_bcond={"slip": True, "adiabatic": True},
        )

        outer_noslip_data = self._generate_boundary_dict(
            name="outer_noslip",
            default_element_size=self.element_size / 2.0,
            default_bcond={"noslip": True, "adiabatic": True},
        )
        nozzle_noslip_data = self._generate_boundary_dict(
            name="nozzle_noslip",
            default_element_size=self.element_size / 8.0,
            default_bcond={"noslip": True, "adiabatic": True},
        )
        nozzle_free_data = self._generate_boundary_dict(
            name="nozzle_free",
            default_element_size=self.element_size / 8.0,
            default_bcond={"normal_force": None, "adiabatic": True},
        )
        symmetry_plane_data = self._generate_boundary_dict(
            name="symmetry_plane",
            default_element_size=self.element_size,
            default_bcond={"slip": True, "adiabatic": True},
        )
        actuator_data = self._generate_boundary_dict(
            name="actuator",
            default_element_size=self.element_size / 20.0,
            default_bcond={"inflow": (0.0, 0.0), "adiabatic": True},
        )
        return (
            outer_slip_data,
            outer_noslip_data,
            nozzle_noslip_data,
            nozzle_free_data,
            symmetry_plane_data,
            actuator_data,
        )

    @property
    def control_boundary(self):
        return self.domain.boundary_elements[-1]

    @property
    def shared_boundary(self):
        return self.domain.boundary_elements[-3]
