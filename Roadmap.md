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

1. Generic approach for DirichletBC generation. In `boundary_components`:
- dolfin.Constant shouldn't be used. It can be int/float, dolfin.Expression or dolfin.Constant. Need to parse this data properly.
- Can the parsing be a BoundaryCondition class method?

#### Major tasks

1. Implement variational forms for different problem types. 


### Optimization modules