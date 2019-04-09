import pytest
from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain


@pytest.fixture
def simple_domain():
    control_points = [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, -0.1],
        [0.3, 0.2],
        [0.4, 0.0],
        [0.5, 0.1],
    ]
    boundary2 = LineElement("type", [[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]])
    boundary1 = BSplineElement("type", control_points)
    domain = SimpleDomain([boundary1, boundary2])
    return domain

def test_create_domain(simple_domain):
    domain = simple_domain
    assert domain.pg_geometry
    assert domain.mesh
    assert len(domain.boundary_elements) == 2