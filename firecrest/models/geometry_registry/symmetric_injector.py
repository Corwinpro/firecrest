from .base_assembler import BaseAssembler


class SymmetricInjectorAssembler(BaseAssembler):
    def generate_control_points(self):
        channel_height = self.geometry_data["channel_height"]
        width = self.geometry_data["channel_width"]
        actuator_length = self.geometry_data["actuator_length"]
        actuator_connector = self.geometry_data["actuator_connector"]
        nozzle_r = self.geometry_data["nozzle_r"]
        nozzle_l = self.geometry_data["nozzle_l"]
        nozzle_offset = self.geometry_data["nozzle_offset"]

        control_points = []

        cp_connector_wall = [
            [width - actuator_length, channel_height - 1.0e-16],
            [width - actuator_length - actuator_connector, channel_height + 1.0e-16],
        ]
        cp_top_left = [
            [width - actuator_length - actuator_connector, channel_height + 1.0e-16],
            [0.0, channel_height],
        ]
        cp_left_side = [[0.0, channel_height], [0.0 - 1.0e-16, 0.0 + 1.0e-16]]
        cp_bot_left = [
            [0.0 - 1.0e-16, 0.0 + 1.0e-16],
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
        cp_symmetry_plane = [[width, -nozzle_l + 1.0e-16], [width, channel_height]]
        cp_actuator = [
            [width, channel_height],
            [width - actuator_length, channel_height - 1.0e-16],
        ]

        control_points.extend(
            [
                cp_connector_wall,
                cp_top_left,
                cp_left_side,
                cp_bot_left,
                cp_nozzle_noslip_wall,
                cp_nozzle_free_surface,
                cp_symmetry_plane,
                cp_actuator,
            ]
        )

        return control_points

    def generate_boundary_metadata(self):
        actuator_connector = self._generate_boundary_dict(
            name="actuator_connector",
            default_element_size=self.element_size,
            default_bcond={"slip": True, "adiabatic": True},
        )
        top_left = self._generate_boundary_dict(
            name="top_left",
            default_element_size=self.element_size,
            default_bcond={"noslip": True, "adiabatic": True},
        )
        left_side = self._generate_boundary_dict(
            name="left_side",
            default_element_size=self.element_size,
            default_bcond={"free": True, "adiabatic": True},
        )
        bot_left = self._generate_boundary_dict(
            name="bot_left",
            default_element_size=self.element_size,
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
            default_element_size=self.element_size / 4.0,
            default_bcond={"inflow": (0.0, 0.0), "adiabatic": True},
        )
        return (
            actuator_connector,
            top_left,
            left_side,
            bot_left,
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
