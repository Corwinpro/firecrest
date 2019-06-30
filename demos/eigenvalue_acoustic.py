from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.fem.tv_acoustic_weakform import (
    ComplexTVAcousticWeakForm,
    TVAcousticWeakForm,
)
import dolfin as dolf


control_points_1 = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
control_points_2 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.05, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.05, bcond={"free": True, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


# domain = dolf.UnitSquareMesh(10, 10)
# domain.mesh = domain
# print(forms.temporal_component())
# print(forms.dirichlet_boundary_conditions())

# forms = ComplexTVAcousticWeakForm(domain, Re=100.0, Pe=1.0)
# a = forms.spatial_component()
# shift = dolf.Constant(2.0) * (
#     forms.temporal_component("real", "imag") - forms.temporal_component("imag", "real")
# )
# b = forms.temporal_component()
# bcs = forms.dirichlet_boundary_conditions()
#
# AA = dolf.PETScMatrix()
# AA = dolf.assemble(-a + shift, tensor=AA)
# for bc in bcs:
#     bc.apply(AA)
# AA = AA.mat()
#
# BB = dolf.PETScMatrix()
# BB = dolf.assemble(b, tensor=BB)
# for bc in bcs:
#     bc.zero(BB)
# BB = BB.mat()
#
# from firecrest.solvers.base_solver import EigenvalueSolver
#
# solver = EigenvalueSolver(domain)
# solver.set_solver_operators(AA, BB)
# solver.solve()
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver

solver = EigenvalueTVAcousticSolver(domain, complex_shift=0 + 2j, Re=100.0, Pe=1.0)
solver.solve()

# ev, rx, cx = solver.retrieve_eigenvalue(0)
# print(ev)
# realParts = dolf.Function(forms.function_space)
# realParts.vector()[:] = rx
# realParts = realParts.split(True)
# imagParts = dolf.Function(forms.function_space)
# imagParts.vector()[:] = cx
# imagParts = imagParts.split(True)
#
# mid = int(len(realParts)/2)
# for j in range(len(realParts)):
#     if j < mid:
#         realParts[j].vector()[:] -= imagParts[j + mid].vector()
#     else:
#         realParts[j].vector()[:] += imagParts[j - mid].vector()
#
#
# _tmp_forms = forms = TVAcousticWeakForm(domain, Re=100.0, Pe=1.0)
#
# realEV = dolf.Function(_tmp_forms.velocity_function_space.collapse())
# imagEV = dolf.Function(_tmp_forms.velocity_function_space.collapse())
# dolf.assign(realEV,realParts[1]); dolf.assign(imagEV,realParts[1])
# file = dolf.File("mode.pvd")
# file << realEV

ev, real_mode, imag_mode = solver.restore_eigenfunction(0)

# realEV = dolf.Function(solver.forms.real_function_space)#.sub(1).collapse())
# dolf.assign(realEV, real_mode)#[1])

file = dolf.File("mode.pvd")
file << real_mode[1], 0
file << imag_mode[1], 1

func = dolf.Function(solver.forms.function_space)
# dolf.assign(func.sub(0), real_mode[0]) # This works
dolf.assign(func, list(real_mode + imag_mode))
file << func.sub(4), 2
