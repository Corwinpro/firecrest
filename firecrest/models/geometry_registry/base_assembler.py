from abc import ABC, abstractmethod

from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain


class BaseAssembler(ABC):
    def __init__(self, geometry_data):
        # Geometry data for acoustic domain
        self.geometry_data = geometry_data

        self._element_size = None
        self._control_points = None
        self._boundary_metadata = None
        self._domain_boundaries = None
        self._domain = None

    @abstractmethod
    def generate_control_points(self):
        pass

    @abstractmethod
    def generate_boundary_metadata(self):
        pass

    def _generate_boundary_dict(self, name, default_element_size, default_bcond):
        boundary_data = self.geometry_data.get(name, {})
        if "el_size" not in boundary_data:
            boundary_data["el_size"] = default_element_size
        if "bcond" not in boundary_data:
            boundary_data["bcond"] = default_bcond
        return boundary_data

    @property
    def element_size(self):
        if self._element_size is None:
            self._element_size = self.geometry_data["element_size"]
        return self._element_size

    @property
    def control_points(self):
        if self._control_points is None:
            self._control_points = self.generate_control_points()

        return self._control_points

    @property
    def boundary_metadata(self):
        if self._boundary_metadata is None:
            self._boundary_metadata = self.generate_boundary_metadata()

        return self._boundary_metadata

    @property
    def domain_boundaries(self):
        if self._domain_boundaries is None:
            self._domain_boundaries = [
                LineElement(control_points, **data)
                for control_points, data in zip(
                    self.control_points, self.boundary_metadata
                )
            ]

        return self._domain_boundaries

    @property
    def domain(self):
        if self._domain is None:
            self._domain = SimpleDomain(self.domain_boundaries)

        return self._domain
