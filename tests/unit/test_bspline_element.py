import pytest
from pysplines.bsplines import Bspline
from firecrest.mesh.boundaryelement import BSplineElement

btype = "boundary_type"
control_points = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
bcond = "boundary_condition"
el_size = 0.1
degree = 3


@pytest.fixture
def bspline():
    bspline = Bspline(control_points, degree=degree)
    bspline.tolerance = 1.0e-5
    return bspline


@pytest.fixture
def bspline_boundary_element():
    boundary = BSplineElement(btype, control_points, bcond, el_size, degree=degree)
    return boundary


def test_create_boundary_element(bspline_boundary_element):
    boundary = bspline_boundary_element
    assert boundary.btype == btype
    assert boundary.control_points == control_points
    assert boundary.bcond == bcond
    assert boundary.el_size == el_size


def test_boundary_surface_index(bspline_boundary_element):
    boundary = bspline_boundary_element
    class_surface_index = BSplineElement.surface_index
    next_boundary = BSplineElement(btype, control_points, bcond, el_size)
    assert next_boundary.surface_index == boundary.surface_index + 1
    assert BSplineElement.surface_index == class_surface_index + 1


def test_boundary_line_list(bspline_boundary_element):
    boundary = bspline_boundary_element
    assert boundary.surface_lines[0][0] == boundary.surface_points[0]
    assert boundary.surface_lines[-1][-1] == boundary.surface_points[-1]
