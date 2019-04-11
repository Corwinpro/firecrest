import pytest
from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain


@pytest.fixture(params=["xml", "xdmf"])
def simple_domain(request):
    control_points = [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, -0.1],
        [0.3, 0.2],
        [0.4, 0.0],
        [0.5, 0.1],
    ]
    boundary1 = BSplineElement("type_one", control_points)
    boundary2 = LineElement(
        "type_two", [[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]]
    )
    domain_boundaries = (boundary1, boundary2)
    domain = SimpleDomain(domain_boundaries, msh_format=request.param)
    return domain


def test_create_domain(simple_domain):
    domain = simple_domain
    assert domain.pg_geometry
    assert domain.mesh
    assert len(domain.boundary_elements) == 2


def test_boundaries_marking(simple_domain):
    assert simple_domain.markers_dict["type_one"] == [
        simple_domain.boundary_elements[0]
    ]
    assert simple_domain.markers_dict["type_two"] == [
        simple_domain.boundary_elements[1]
    ]
