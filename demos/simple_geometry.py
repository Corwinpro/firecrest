from firecrest.mesh.boundaryelement import BSplineElement
from firecrest.mesh.geometry import SimplyConnected

control_points = [
    [0.0, 0.0],
    [0.1, 0.1],
    [0.2, -0.1],
    [0.3, 0.2],
    [0.4, 0.0],
    [0.5, 0.1],
]
boundary2 = BSplineElement(
    "type", [[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]], degree=1
)
boundary1 = BSplineElement("type", control_points)
domain = SimplyConnected([boundary1, boundary2])
