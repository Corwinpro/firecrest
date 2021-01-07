import dolfin as dolf

from .geometry import Geometry
from .boundaryelement import BoundaryElement


class IntervalDomain(Geometry):
    def __init__(self, boundary_elements, **kwargs):
        self.dimensions = 1
        self.boundary_elements = boundary_elements
        self.resolution = kwargs.get("resolution", 100)
        self.start_point = kwargs.get("a", 0.0)
        self.end_point = kwargs.get("b", 1.0)

        self.mark_boundaries()

        self.mesh = dolf.IntervalMesh(self.resolution, self.start_point, self.end_point)
        self._boundary_parts = None

    @property
    def n(self):
        return super().n[0]

    @property
    def boundary_parts(self):
        """
        Creates dolfin MeshFunction of dim-1 dimension
        """
        if self._boundary_parts:
            return self._boundary_parts

        boundary_parts = dolf.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        boundary_parts.set_all(0)
        for boundary in self.boundary_elements:
            boundary.inside_subdomain.mark(boundary_parts, boundary.surface_index)

        self._boundary_parts = boundary_parts
        return boundary_parts

    @property
    def ds(self):
        """
        Creates dolfin 'ds' measure, accounting for physical marking
        """
        assert self.mesh, "Mesh needs to be generated"

        return dolf.Measure("ds", domain=self.mesh, subdomain_data=self.boundary_parts)


class PointBoundary(BoundaryElement):
    def __init__(self, control_points, bcond=None, **kwargs):
        super().__init__(control_points, bcond=bcond, **kwargs)
        inside_rule = kwargs.get("inside")

        class InsideSubdomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return inside_rule(x) and on_boundary

        self.inside_subdomain = InsideSubdomain()

    def generate_boundary(self):
        pass

    def get_normal(self, point):
        pass

    def get_curvature(self, point):
        pass

    def get_displacement(self, point, cp_index):
        pass
