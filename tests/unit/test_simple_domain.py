import pytest
from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain
import dolfin as dolf

"""
Simple domain currently doesn't work with xdmf because of the boundary markings
"""


@pytest.fixture(scope="module", params=["xml"])  # , "xdmf"
def simple_domain(request):
    control_points = [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, -0.1],
        [0.3, 0.2],
        [0.4, 0.0],
        [0.5, 0.1],
    ]
    boundary1 = BSplineElement(control_points, bcond={"type_one": True})
    boundary2 = LineElement(
        [[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]], bcond={"type_two": True}
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


def test_individual_dolfin_measures(simple_domain):
    assert (
        abs(
            dolf.assemble(dolf.Constant(1.0) * simple_domain.ds()) ** 2.0
            - 1.5631 ** 2.0
        )
        < 0.1 ** 2.0
    )
    assert (
        abs(dolf.assemble(dolf.Constant(1.0) * simple_domain.dx) ** 2.0 - 0.1245 ** 2.0)
        < 0.1 ** 2.0
    )

    assert (
        abs(
            dolf.assemble(
                dolf.Constant(1.0)
                * simple_domain.ds((simple_domain.boundary_elements[0].surface_index,))
            )
            ** 2.0
            - 0.587 ** 2.0
        )
        < 0.1 ** 2.0
    )
    assert (
        abs(
            dolf.assemble(
                dolf.Constant(1.0)
                * simple_domain.ds((simple_domain.boundary_elements[1].surface_index,))
            )
            ** 2.0
            - 0.976 ** 2.0
        )
        < 0.1 ** 2.0
    )


def test_groups_dolfin_measures(simple_domain):
    assert (
        abs(
            dolf.assemble(
                dolf.Constant(1.0) * simple_domain.get_boundary_measure("type_one")
            )
            ** 2.0
            - 0.587 ** 2.0
        )
        < 0.1 ** 2.0
    )
    assert (
        abs(
            dolf.assemble(
                dolf.Constant(1.0) * simple_domain.get_boundary_measure("type_two")
            )
            ** 2.0
            - 0.976 ** 2.0
        )
        < 0.1 ** 2.0
    )


def test_get_boundaries(simple_domain):
    assert simple_domain._get_boundaries("type_one") == [
        simple_domain.boundary_elements[0]
    ]
    assert simple_domain._get_boundaries("type_two") == [
        simple_domain.boundary_elements[1]
    ]
    with pytest.raises(KeyError):
        simple_domain._get_boundaries("type_three")
