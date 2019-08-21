## Main project stages

### Geometry and mesh modules

#### Major tasks

1. Implement load and export geometry and mesh objects. 
It is much more efficient and convenient to reuse them, rather then build every time.

#### Minor tasks

1. Geometry to JSON format: reusable geometry definition from templates.
2. Implementing special boundary elements: arc, ellipse.
3. 3D compatibility.

### FEM modules

#### Current task

1. Eigensolver: implement the surface tension BC
2. Should the Boundary conditions be in a separate class?

#### Major tasks

1. Weak forms: write tests for `real` / `complex` usage

#### Minor tasks


### Optimization modules

1.