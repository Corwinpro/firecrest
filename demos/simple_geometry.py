from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain

control_points = [
    [0.0, 0.0],
    [0.1, 0.1],
    [0.2, -0.1],
    [0.3, 0.2],
    [0.4, 0.0],
    [0.5, 0.1],
]
boundary2 = LineElement([[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]])
boundary1 = BSplineElement(control_points)
domain = SimpleDomain([boundary1, boundary2])
