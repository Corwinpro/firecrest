## Main project stages

### Geometry and mesh modules

#### Major tasks

1. Find a good approach and implement BoundaryConditions.
2. Integration with solver module.

#### Minor tasks

1. Geometry to JSON format: reusable geometry definition from templates.
2. Implementing special boundary elements: arc, ellipse.
3. 3D compatibility.

### FEM modules

#### Current task

1. The method 'TVAcousticWeakForm().boundary_components' should parse the 'markers_dict' object of it's domain, not each boundary separately.
2. Boundary conditions should be in a separate class. I still don't understand how to couple a boundary condition with weak form classes.

#### Major tasks

1. Implement variational forms for different problem types. 

#### Minor tasks


### Optimization modules