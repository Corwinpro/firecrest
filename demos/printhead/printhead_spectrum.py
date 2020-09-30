"""
Eingenvalue problem for a symmetric inkjet printhead geometry.

Two symmetric systems can be considered:
    - with slip boundary on the symmetry plane,
        This results in odd eigenmodes
    - with stress-free boundary on the symmetry plane.
        This results in even eigenmodes
"""
from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
import dolfin as dolf
import ufl
import matplotlib.pyplot as plt

control_points_0 = [[9.1, -0.2], [9.1, 0.0]]
control_points_1 = [[9.1, 0.0], [0.0, 0.0], [0.0, 4.7]]
control_points_2 = [[0.0, 4.7], [2.0, 4.7]]
control_points_3 = [[2.0, 4.7], [2.0, 0.7], [9.2, 0.7]]
control_points_4 = [[9.2, 0.7], [9.2, -0.2]]
control_points_5 = [[9.2, -0.2], [9.1, -0.2]]


el_size = 0.03
boundary0 = LineElement(
    control_points_0, el_size=el_size / 4.0, bcond={"noslip": True, "isothermal": True}
)
boundary1 = LineElement(
    control_points_1, el_size=el_size, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=el_size, bcond={"free": True, "adiabatic": True}
)
boundary3 = LineElement(
    control_points_3, el_size=el_size, bcond={"noslip": True, "isothermal": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=el_size / 2.0,
    bcond={"slip": True, "adiabatic": True},  # symmetric
    # bcond={"free": True, "isothermal": True},  # antisymmetric
)
boundary5 = LineElement(
    control_points_5, el_size=el_size / 4.0, bcond={"noslip": True, "adiabatic": True}
)
domain_boundaries = (boundary0, boundary1, boundary2, boundary3, boundary4, boundary5)
domain = SimpleDomain(domain_boundaries)

solver = EigenvalueTVAcousticSolver(
    # domain, complex_shift=-0.03 + 0.8j, Re=6000.0, Pe=1.0, nmodes=16
    domain,
    complex_shift=-0.004 + 0.13j,
    Re=6000.0,
    Pe=1.0,
    nmodes=2,
)
solver.solve()

spectrum = []


for i in range(int(solver.nof_modes_converged / 2)):
    ev, real_mode, imag_mode = solver.extract_solution(i)
    spectrum.append(ev)
    solver.output_field(real_mode + imag_mode)

    i, j = ufl.indices(2)
    energy_real = dolf.assemble(solver.forms.temporal_component(real_mode, real_mode))
    energy_imag = dolf.assemble(solver.forms.temporal_component(real_mode, real_mode))
    energy = energy_real + energy_imag * 1.0j
    energy = abs(energy)

    f = dolf.File("Visualization/sigma_real.pvd")
    pressure, velocity, temperature = real_mode
    continuity_component = pressure * dolf.div(velocity)
    momentum_component = dolf.inner(
        dolf.as_tensor(velocity[i].dx(j), (i, j)),
        solver.forms.stress(pressure, velocity),
    )
    energy_component = dolf.inner(
        dolf.grad(temperature), solver.forms.heat_flux(temperature)
    )
    st = dolf.project(
        (continuity_component + momentum_component + energy_component) / energy,
        dolf.FunctionSpace(solver.domain.mesh, "CG", 2),
    )
    st.rename("sigma_real", "sigma_real")
    f << st

    f = dolf.File("Visualization/sigma_imag.pvd")
    pressure, velocity, temperature = imag_mode
    continuity_component = pressure * dolf.div(velocity)
    momentum_component = dolf.inner(
        dolf.as_tensor(velocity[i].dx(j), (i, j)),
        solver.forms.stress(pressure, velocity),
    )
    energy_component = dolf.inner(
        dolf.grad(temperature), solver.forms.heat_flux(temperature)
    )
    st = dolf.project(
        (continuity_component + momentum_component + energy_component) / energy,
        dolf.FunctionSpace(solver.domain.mesh, "CG", 2),
    )
    st.rename("sigma_imag", "sigma_imag")
    f << st


plt.plot([ev.real for ev in spectrum], [ev.imag for ev in spectrum], "o")
for value in spectrum:
    print(f"{value.real}, {value.imag}")


short_spectrum = [
    (-0.029796769962597662 + 0.35294042262432046j),
    (-0.029512514468998545 + 0.5581814336549202j),
    (-0.040852252316042476 + 0.15539180023941337j),
    (-0.03360330688324703 + 0.7822368881240528j),
]
plt.plot([ev.real for ev in short_spectrum], [ev.imag for ev in short_spectrum], "x")

plt.grid(True)
plt.show()
