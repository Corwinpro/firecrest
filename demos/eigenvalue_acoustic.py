from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.fem.tv_acoustic_fspace import TVAcousticFunctionSpace
import dolfin as dolf

"""
control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.1, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.1, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)
"""

domain = dolf.UnitSquareMesh(10, 10)
domain.mesh = domain
function_space_factory = TVAcousticFunctionSpace(domain)
function_space = function_space_factory.function_spaces
trial_functions = dolf.TrialFunctions(function_space)
test_functions = dolf.TestFunctions(function_space)
print(*trial_functions, sep="\n")

