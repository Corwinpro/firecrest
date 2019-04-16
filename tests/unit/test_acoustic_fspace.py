import pytest
from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.fem.tv_acoustic_fspace import TVAcousticFunctionSpace


@pytest.fixture(scope="module")
def simple_domain():
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
    domain = SimpleDomain(domain_boundaries)
    return domain


def test_create_acoustic_fspace(simple_domain):
    function_space = TVAcousticFunctionSpace(simple_domain)
    assert function_space
    assert function_space.function_spaces


def test_create_complex_acoustic_fspace(simple_domain):
    function_space = TVAcousticFunctionSpace(simple_domain, is_complex=True)
    assert function_space
    assert function_space.function_spaces
