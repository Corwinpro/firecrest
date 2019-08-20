from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.spectral_tv_acoustic_solver import SpectralTVAcousticSolver

control_points_1 = [[1.0e-16, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
control_points_2 = [[1.0, 1.0], [1.0e-16, 1.0]]
boundary1 = LineElement(
    control_points_1, el_size=0.03, bcond={"noslip": True, "isothermal": True}
)
boundary2 = LineElement(
    control_points_2, el_size=0.03, bcond={"normal_velocity": 1.0, "adiabatic": True}
)
domain_boundaries = (boundary1, boundary2)
domain = SimpleDomain(domain_boundaries)


solver = SpectralTVAcousticSolver(domain, frequency=10.5j, Re=500.0, Pr=1.0)
state = solver.solve()
solver.output_field(state)
