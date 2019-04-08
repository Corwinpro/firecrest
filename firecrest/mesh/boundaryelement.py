from abc import ABC, abstractmethod
from pysplines.bsplines import Bspline
import numpy as np


def on_surface(surface_lines, x):
    tolerance = 1.0e-5
    for line in surface_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]

        if x1 > x2:
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)

        between_extremities = x[0] >= x1 and x[0] <= x2
        is_vertical_line = x1 == x2
        is_online = False

        if is_vertical_line:
            if y1 > y2:
                is_online = x[1] >= y2 and x[1] <= y1
            else:
                is_online = x[1] <= y2 and x[1] >= y1
        else:
            is_online = abs(y1 - x[1] + (x[0] - x1) * (y2 - y1) / (x2 - x1)) < tolerance

        if between_extremities and is_online:
            return True

    return False


class BoundaryElement(ABC):
    """
    Generic class for Boundary Elements. 
    A Boundary Element represents a piece of a domain surface with 
    certain properties, i.e. a physical boundary type (no slip boundary), 
    a geometric representation (B-spline or circular arc).

    params:
        - btype : boundary type defined by boundary conditions
        - control_points : boundary parametrization by control points
        - bcond : specific boundary condition
        - el_size : characteristic size of the line elements on the surface

    attributes:
        - surface_index: when we create a new surface, a unique index is assigned
        to it, so we can track the individual properties of the boundary elements
    """

    surface_index = 1

    def __init__(self, btype, control_points, bcond=None, el_size=0.05, **kwargs):
        self.btype = btype
        self._control_points = control_points
        self.bcond = bcond
        self.el_size = el_size
        self.kwargs = kwargs

        self.boundary = None
        self.surface_index = BoundaryElement.surface_index
        BoundaryElement.surface_index += 1

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, control_points):
        self._control_points = control_points
        self.generate_boundary()

    @abstractmethod
    def generate_boundary(self):
        self.surface_points = []
        self.surface_lines = []

    @abstractmethod
    def get_normal(self, point):
        pass

    @abstractmethod
    def get_curvature(self, point):
        pass

    @abstractmethod
    def get_displacement(self, point, cp_index):
        pass

    @staticmethod
    def _create_line_list(point_list):
        line_list = [
            [point_list[i], point_list[i + 1]] for i in range(len(point_list) - 1)
        ]
        return line_list

    @staticmethod
    def estimate_points_number(points, el_size):
        p = np.array(points)
        _dist = sum([np.linalg.norm(p[i] - p[i + 1]) for i in range(len(p) - 1)])
        n_of_points = int(_dist / el_size) + 1
        return n_of_points


class BSplineElement(BoundaryElement):
    def __init__(self, btype, control_points, bcond=None, el_size=0.05, **kwargs):
        super().__init__(btype, control_points, bcond=bcond, el_size=el_size, **kwargs)
        self.spline_degree = kwargs.pop("degree", 3)
        self.spline_periodic = kwargs.pop("periodic", False)
        self.n = self.estimate_points_number(self.control_points, self.el_size)
        self.kwargs = kwargs

        self.control_points = control_points

    def generate_boundary(self):
        super().generate_boundary()
        self.boundary = Bspline(
            self.control_points,
            self.spline_degree,
            n=self.n,
            periodic=self.spline_periodic,
            **self.kwargs
        )
        self.surface_points = self.boundary.rvals
        self.surface_points[0] = self.control_points[0]
        self.surface_points[-1] = self.control_points[-1]
        self.surface_lines = self._create_line_list(self.surface_points)

    def get_normal(self, point):
        return self.boundary.normal(point)

    def get_curvature(self, point):
        return self.boundary.curvature(point)

    def get_displacement(self, point, cp_index):
        return self.boundary.displacement(point, cp_index)


class LineElement(BSplineElement):
    def __init__(self, btype, control_points, bcond=None, el_size=0.05, **kwargs):
        if (
            np.linalg.norm(np.array(control_points[0] - np.array(control_points[-1])))
            < 1.0e-10
        ):
            periodic = True
        else:
            periodic = False
        super().__init__(
            btype,
            control_points,
            bcond=bcond,
            el_size=el_size,
            degree=1,
            periodic=periodic,
            **kwargs
        )

