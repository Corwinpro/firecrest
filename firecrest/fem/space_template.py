from collections import namedtuple

"""
Space is a template tuple that defines Finite Element functions space:
- element_type: CG, DG, etc.
- order: element polynomial order P0, P1, P2, etc.
- dimension: the dimensionality of the element: 
    0,1, 'scalar' - for scalar
    n > 1 - for n-dimensional vector
    'vector' - for vector of the mesh dimensionality

We will use this structure to create necessary Function Spaces for pre-defined
problems, such as thermoviscous flow, etc.
"""
Space = namedtuple("Space", ["element_type", "order", "dimension"])
