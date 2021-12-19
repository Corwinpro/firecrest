from .base_assembler import BaseAssembler


class SymmetricPrintheadGeometryAssembler(BaseAssembler):
    def generate_control_points(self):
        channel_height = self.geometry_data["channel_height"]
        length = self.geometry_data["channel_length"]
        actuator_length = self.geometry_data["actuator_length"]
        nozzle_r = self.geometry_data["nozzle_r"]
        nozzle_l = self.geometry_data["nozzle_l"]
        nozzle_offset = self.geometry_data["nozzle_offset"]
        manifold_width = self.geometry_data["manifold_width"]
        manifold_height = self.geometry_data["manifold_height"]

        cp_inner_noslip_wall = [
            [length - actuator_length, channel_height],
            [manifold_width, channel_height],
            [manifold_width, manifold_height],
        ]
        cp_manifold_free_surface = [
            [manifold_width, manifold_height],
            [1.0e-16, manifold_height],
        ]
        cp_outer_noslip_wall = [
            [1.0e-16, manifold_height],
            [0.0, 0.0],
            [length - nozzle_r - nozzle_offset, 0.0],
        ]
        cp_nozzle_noslip_wall = [
            [length - nozzle_r - nozzle_offset, 0.0],
            [length - nozzle_r, 0.0],
            [length - nozzle_r, -nozzle_l],
        ]
        cp_nozzle_free_surface = [
            [length - nozzle_r, -nozzle_l],
            [length, -nozzle_l + 1.0e-16],
        ]
        cp_symmetry_plane = [
            [length, -nozzle_l + 1.0e-16],
            [length, channel_height + 1.0e-16],
        ]
        cp_actuator = [
            [length, channel_height + 1.0e-16],
            [length - actuator_length, channel_height],
        ]

        control_points = [
            cp_inner_noslip_wall,
            cp_manifold_free_surface,
            cp_outer_noslip_wall,
            cp_nozzle_noslip_wall,
            cp_nozzle_free_surface,
            cp_symmetry_plane,
            cp_actuator,
        ]
        return control_points

    def generate_boundary_metadata(self):
        inner_noslip_wall_data = self._generate_boundary_dict(
            name="inner_noslip_meta",
            default_element_size=self.element_size,
            default_bcond={"noslip": True, "adiabatic": True},
        )
        manifold_free_surface_data = self._generate_boundary_dict(
            name="manifold_free_meta",
            default_element_size=self.element_size * 2.0,
            default_bcond={"free": True, "adiabatic": True},
        )
        outer_noslip_data = self._generate_boundary_dict(
            name="outer_noslip_meta",
            default_element_size=self.element_size,
            default_bcond={"noslip": True, "adiabatic": True},
        )
        nozzle_noslip_data = self._generate_boundary_dict(
            name="nozzle_noslip_meta",
            default_element_size=self.element_size / 8.0,
            default_bcond={"noslip": True, "adiabatic": True},
        )
        nozzle_free_data = self._generate_boundary_dict(
            name="nozzle_free_meta",
            default_element_size=self.element_size / 8.0,
            default_bcond={"normal_force": None, "adiabatic": True},
        )
        symmetry_plane_data = self._generate_boundary_dict(
            name="nozzle_free_meta",
            default_element_size=self.element_size / 2.0,
            default_bcond={"slip": True, "adiabatic": True},
        )
        actuator_data = self._generate_boundary_dict(
            name="actuator_meta",
            default_element_size=self.element_size / 20.0,
            default_bcond={"inflow": (0.0, 0.0), "adiabatic": True},
        )
        return (
            inner_noslip_wall_data,
            manifold_free_surface_data,
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
