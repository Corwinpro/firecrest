"""
Helpful functions to manipulate with dolfin.Vectors and PETSc matrices.
The functions are used for manual sparse matrix constructions, i.e. the
nonlocal weak forms.
"""

import dolfin as dolf
import numpy as np
from petsc4py import PETSc


def boundary_dofs(boundary_parts, boundary, space, boundary_condition):
    """
    Returns indices of degrees of freedom on the given boundary

    :param boundary_parts: domain boundary_parts
    :param boundary: boundary element
    :param space: the space to find the degrees of freedom of
    :param boundary_condition: placeholder boundary condition
    :return: list of dof indices
    """
    # This might be another way to do it:
    # d2v = dolf.dof_to_vertex_map(simple_space)
    # vertices_on_boundary = d2v[simple_function.vector() == 1]
    # for v in vertices_on_boundary:
    #     print(solver.domain.mesh.coordinates()[v])

    bc = dolf.DirichletBC(
        space, boundary_condition, boundary_parts, boundary.surface_index
    )
    return list(bc.get_boundary_values().keys())


def vector_to_ndarray(vec, dofs):
    """
    Convert 'dolfin.cpp.la.Vector' into 'numpy.ndarray'
    :param vec: dolfin.cpp.la.Vector
    :param dofs: list of degrees of freedom indices
    :return: numpy array of (index, local value)
    """
    vec = vec.get_local()
    array = []
    for i in dofs:
        if abs(vec[i]) > 1.0e-16:
            array.append([i, vec[i]])
    return np.array(array)


def outer_to_matrix(full_space, trial_vector, test_vector):
    matrix_dim = len(full_space.tabulate_dof_coordinates())
    block_dim = len(trial_vector) * len(test_vector)

    row = np.zeros(block_dim)
    col = np.zeros(block_dim)
    val = np.zeros(block_dim)

    for i in range(len(test_vector)):
        for j in range(len(trial_vector)):
            col[i * len(trial_vector) + j] = trial_vector[j, 0]
            row[i * len(trial_vector) + j] = test_vector[i, 0]
            val[i * len(trial_vector) + j] = test_vector[i, 1] * trial_vector[j, 1]

    row = row.astype(dtype="int32")
    col = col.astype(dtype="int32")

    indptr = np.bincount(row, minlength=matrix_dim)
    indptr = np.insert(indptr, 0, 0).cumsum()
    indptr = indptr.astype(dtype="int32")

    A = PETSc.Mat().createAIJ(size=(matrix_dim, matrix_dim), csr=(indptr, col, val))
    return A
