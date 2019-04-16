import pytest
from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.fem.base_fspace import BaseFunctionSpace
from firecrest.fem.space_template import Space
import dolfin as dolf


@pytest.fixture(scope="module")
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
    domain = SimpleDomain(domain_boundaries)
    return domain


def test_space_template():
    space = Space(element_type="type", order=2, dimension=3)
    assert space.element_type == "type"
    assert space.order == 2
    assert space.dimension == 3


@pytest.mark.parametrize(
    "space",
    [
        (Space(element_type="CG", order=2, dimension=2),),
        (Space(element_type="DG", order=3, dimension="scalar"),),
        (
            Space(element_type="CG", order=2, dimension="vector"),
            Space(element_type="DG", order=3, dimension="scalar"),
        ),
    ],
)
def test_create_single_space(space, simple_domain):
    function_space = BaseFunctionSpace(simple_domain, space)
    assert function_space
    assert function_space.function_spaces
