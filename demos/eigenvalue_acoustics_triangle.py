from firecrest.api import LineElement, SimpleDomain, EigenvalueTVAcousticSolver

z = -1.0 - 2.0j
control_points_1 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
control_points_2 = [[1.0, 1.0], [0.0, 0.0]]
boundary_1 = LineElement(control_points_1, bcond={"noslip": True, "isothermal": True})
boundary_2 = LineElement(control_points_2, bcond={"impedance": z, "adiabatic": True})
domain = SimpleDomain((boundary_1, boundary_2))

solver = EigenvalueTVAcousticSolver(domain, complex_shift=-3.0j, Re=100.0, Pr=1.0)

if __name__ == "__main__":

    (ev, real_mode, imag_mode), *_ = solver.solve(number_of_modes=1)
    solver.output_field(real_mode + imag_mode)
    print(solver.forms.mass_flow_rate(real_mode, boundary_1))
