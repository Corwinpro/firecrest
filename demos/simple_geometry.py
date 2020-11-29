from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain

line_control_points = [[0.5, 0.1], [0.5, -0.2], [0.0, -0.2], [0.0, 0.0]]

bspline_control_points = [
    [0.0, 0.0],
    [0.1, 0.1],
    [0.2, -0.1],
    [0.3, 0.2],
    [0.4, 0.0],
    [0.5, 0.1],
]

if __name__ == "__main__":
    bspline_boundary = BSplineElement(bspline_control_points)
    line_boundary = LineElement(line_control_points, el_size=0.2)
    domain = SimpleDomain([bspline_boundary, line_boundary])
