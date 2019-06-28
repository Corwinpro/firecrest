import pytest
from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.fem.tv_acoustic_weakform import TVAcousticWeakForm


@pytest.fixture(scope="module")
def forms():
    control_points_1 = [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, -0.1],
        [0.3, 0.2],
        [0.4, 0.0],
        [0.5, 0.1],
    ]
    control_points_2 = [[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]]
    boundary1 = BSplineElement(
        control_points_1, bcond={"noslip": True, "heat_flux": 10}
    )
    boundary2 = LineElement(
        control_points_2, bcond={"impedance": 5, "isothermal": True}
    )
    domain_boundaries = (boundary1, boundary2)
    domain = SimpleDomain(domain_boundaries)
    forms = TVAcousticWeakForm(domain, Re=1, Pe=1)
    return forms


def test_create_spatial_forms(forms):
    assert forms.spatial_component()


def test_create_temporal_forms(forms):
    assert forms.temporal_component()


def test_create_boundary_forms(forms):
    stress_bcomponents, temp_bcomponents = forms.boundary_components()
    assert stress_bcomponents
    assert temp_bcomponents
    bcs = forms.dirichlet_boundary_conditions()
    assert len(bcs) == 2
