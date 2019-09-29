from firecrest.mesh.boundaryelement import LineElement
from firecrest.mesh.geometry import SimpleDomain
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver
from firecrest.models.free_surface_cap import SurfaceModel, AdjointSurfaceModel
import dolfin as dolf
from collections import namedtuple
from decimal import Decimal
from firecrest.misc.time_storage import TimeSeries, PiecewiseLinearBasis
from firecrest.misc.optimization_mixin import OptimizationMixin
import numpy as np

elsize = 0.05
height = 0.7
length = 9.2
actuator_length = 4.0
offset_top = 1.0
nozzle_r = 0.1
nozzle_l = 0.2
nozzle_offset = 0.2
manifold_width = 2.0
manifold_height = 4.7

# Define constants
L = 1.0e-4
c_s = 1.0e3
rho = 1.0e3
epsilon = 1.0e-3
gamma_st = 50.0e-3
mu = 1.0e-2
Re = rho * c_s * L / mu
a1 = 4.0 / 3.0
a2 = 8
kappa_prime = epsilon / nozzle_r ** 4 * 0.25  # up to a constant

# non dimensional
alpha_1 = nozzle_l / (3.1415 * nozzle_r ** 2) * a1
alpha_2 = nozzle_l / (3.1415 * nozzle_r ** 3) / Re * a2 / nozzle_r
gamma_nd = 2 * gamma_st / (rho * c_s ** 2 * nozzle_r * epsilon)


Constants = namedtuple(
    "Constants",
    [
        "density",
        "acoustic_mach",
        "channel_L",
        "nozzle_L",
        "nozzle_R",
        "surface_tension",
        "Re",
        "sound_speed",
    ],
)
printhead_constants = Constants(rho, epsilon, L, 10.0e-6, 10.0e-6, gamma_st, Re, c_s)

nondim_constants = Constants(
    1.0,
    printhead_constants.acoustic_mach,
    1,
    printhead_constants.nozzle_L / printhead_constants.channel_L,
    printhead_constants.nozzle_R / printhead_constants.channel_L,
    2.0
    * printhead_constants.surface_tension
    / (
        printhead_constants.density
        * printhead_constants.sound_speed ** 2.0
        * printhead_constants.nozzle_R
        * printhead_constants.acoustic_mach
    ),
    printhead_constants.Re,
    1.0,
)

control_points_1 = [
    [length - actuator_length, height],
    [manifold_width, height],
    [manifold_width, manifold_height],
]
control_points_free_left = [
    [manifold_width, manifold_height],
    [1.0e-16, manifold_height],
]
control_points_bot_left = [
    [1.0e-16, manifold_height],
    [0.0, 0.0],
    [length - nozzle_r - nozzle_offset, 0.0],
]
control_points_refine_left = [
    [length - nozzle_r - nozzle_offset, 0.0],
    [length - nozzle_r, 0.0],
    [length - nozzle_r, -nozzle_l],
]
control_points_2 = [[length - nozzle_r, -nozzle_l], [length, -nozzle_l + 1.0e-16]]
control_points_3 = [[length, -nozzle_l + 1.0e-16], [length, height + 1.0e-16]]
control_points_4 = [[length, height + 1.0e-16], [length - actuator_length, height]]


boundary1 = LineElement(
    control_points_1, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_free_left = LineElement(
    control_points_free_left,
    el_size=elsize * 2.0,
    bcond={"free": True, "adiabatic": True},
)
boundary_bot_left = LineElement(
    control_points_bot_left, el_size=elsize, bcond={"noslip": True, "adiabatic": True}
)
boundary_refine_left = LineElement(
    control_points_refine_left,
    el_size=elsize / 4.0,
    bcond={"noslip": True, "adiabatic": True},
)
boundary2 = LineElement(
    control_points_2,
    el_size=elsize / 4.0,
    bcond={"normal_force": None, "adiabatic": True},
)
boundary3 = LineElement(
    control_points_3, el_size=elsize / 2.0, bcond={"slip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=elsize / 10.0,
    bcond={"inflow": (0.0, 0.0), "adiabatic": True},
)
domain_boundaries = (
    boundary1,
    boundary_free_left,
    boundary_bot_left,
    boundary_refine_left,
    boundary2,
    boundary3,
    boundary4,
)
domain = SimpleDomain(domain_boundaries)


class NormalInflow:
    def __init__(self, series: TimeSeries):
        self.counter = 1
        self.series = list(series.values())

    def eval(self):
        if self.counter < len(self.series):
            value = (0.0, self.series[self.counter])
            self.counter += 1
            return value
        return 0.0, 0.0


class OptimizationSolver(OptimizationMixin, UnsteadyTVAcousticSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_basis = PiecewiseLinearBasis(
            np.array([float(key) for key in default_grid.keys()]),
            width=kwargs.get("signal_window", 2.0),
        )

    def flow_rate(self, state):
        return 2.0 * dolf.assemble(
            dolf.inner(state[1], self.domain.n)
            * self.domain.ds((boundary2.surface_index,))
        )

    def _objective_state(self, control):
        restored_control = self.linear_basis.extrapolate([0.0] + list(control) + [0.0])
        boundary4.bcond["inflow"] = NormalInflow(
            TimeSeries.from_list(restored_control, default_grid)
        )

        surface_model = SurfaceModel(nondim_constants, kappa_t0=0.25)
        boundary2.bcond["normal_force"] = surface_model

        _old_flow_rate = 0
        for state in self.solve_direct(initial_state, verbose=False, yield_state=True):
            if isinstance(state, TimeSeries):
                direct_history = state
                break
            _new_flow_rate = self.flow_rate(state)
            surface_model.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )
            _old_flow_rate = _new_flow_rate

        with open("log.dat", "a") as file:
            file.write(
                str(
                    self._objective((direct_history.last, surface_model), verbose=False)
                )
                + "  "
                + str(surface_model.kappa)
                + ":\t"
            )
            for c in control:
                file.write(str(c) + ",")
            file.write("\n")

        return direct_history.last, surface_model

    def _objective(self, state, verbose=True):
        if verbose:
            print(
                "evaluated at: ",
                self.forms.energy(state[0])
                + state[1].surface_energy(state[1].kappa) / 2.0,
                " curvature: ",
                state[1].kappa,
            )
        return (
            self.forms.energy(state[0]) + state[1].surface_energy(state[1].kappa) / 2.0
        )

    def _jacobian(self, state):
        state, surface_model = state
        state = (state[0], -state[1], state[2])

        adjoint_surface = AdjointSurfaceModel(direct_surface=surface_model)
        boundary2.bcond["normal_force"] = adjoint_surface

        _old_flow_rate = 0
        _new_flow_rate = 0
        adjoint_surface.update_curvature(
            0.5 * (_new_flow_rate + _old_flow_rate), self._dt
        )
        for adjoint_state in solver.solve_adjoint(
            initial_state=state, verbose=False, yield_state=True
        ):
            if isinstance(adjoint_state, TimeSeries):
                adjoint_history = adjoint_state
                break
            _old_flow_rate = _new_flow_rate
            _new_flow_rate = self.flow_rate(adjoint_state)
            adjoint_surface.update_curvature(
                0.5 * (_new_flow_rate + _old_flow_rate), self._dt
            )

        adjoint_stress = adjoint_history.apply(lambda x: self.forms.stress(x[0], x[1]))
        adjoint_stress_averaged = adjoint_stress.apply(
            lambda x: dolf.assemble(
                dolf.dot(dolf.dot(x, self.domain.n), self.domain.n)
                * self.domain.ds((boundary4.surface_index,))
            )
        )

        du = TimeSeries.interpolate_to_keys(adjoint_stress_averaged, small_grid)
        du[Decimal("0")] = 0.0
        du[timer["T"]] = 0.0

        discrete_grad = self.linear_basis.discretize(du.values())
        discrete_grad[0] = 0.0
        discrete_grad[-1] = 0.0

        print("gradient norm: ", (adjoint_stress_averaged * du).integrate())

        print(
            "discrete gradient norm: ",
            (
                adjoint_stress_averaged
                * TimeSeries.from_list(
                    self.linear_basis.extrapolate(discrete_grad), default_grid
                )
            ).integrate(),
        )

        return discrete_grad[1:-1]


timer = {"dt": Decimal("0.01"), "T": Decimal("20.0")}
default_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)
small_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(1, int(timer["T"] / Decimal(timer["dt"])))
    }
)

surface_model = SurfaceModel(nondim_constants, kappa_t0=0.25)
solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer, signal_window=0.5)
initial_state = (0.0, (0.0, 0.0), 0.0)

coarse_space_control = [
    0.0031218434670749083,
    0.00039993341910226596,
    -0.0008878779110163696,
    -0.0008273270413055289,
    0.0006028534180741699,
    0.0030869918808126085,
    0.0021383126783835207,
    0.0003107608476460111,
    0.0004944440592362955,
    0.00022870287117476664,
    0.0005532036274521335,
    7.096444090708459e-05,
    0.002310475217827071,
    0.0028336197709607387,
    0.00028401286298087127,
    -0.0009931305927794536,
    -0.0005059536985463389,
    0.0005782996337251334,
    0.0029616268344061186,
]
fine_space_control = [
    0.005631493573713741,
    0.00360726062923221,
    0.0020760121966716494,
    0.0015564167290605736,
    0.001277809866803225,
    -0.001494795389589515,
    -0.002596232661207128,
    -0.0005200884519516501,
    -8.684405427640264e-05,
    0.00010345197663807958,
    0.00027091359929422114,
    -0.0026438361759351723,
    -0.001344683618805592,
    -0.0007617045594648097,
    -0.0007380331956165771,
    -0.0008159421636108862,
    8.69443510343832e-05,
    -0.00017403307021087987,
    0.0011859156281541457,
    0.002246500481689057,
    0.0023685025692204618,
    0.0019068535831057256,
    0.0014808449466082087,
    0.0013343736589892287,
    0.0009289089807176356,
    0.0009380499609701089,
    0.0013514482891215714,
    0.0023612407413943744,
    0.0008738138206758879,
    -0.0003669870213387837,
    -0.0008493750078432896,
    -0.00025584916714956117,
    0.0004759618773532267,
    0.0011960764354408354,
    0.0011476771411034357,
    0.00021241589244945825,
    0.000533182619766068,
    3.5217003196981733e-05,
    -0.00015678823559354767,
    4.6981497791060796e-05,
    -0.000614503255875759,
    5.1994427723881775e-05,
    -1.83417954841954e-05,
    -3.8388102432740064e-05,
    0.0011859791275266136,
    0.0012922722603821398,
    0.0007967812255950538,
    -0.00014291395918308847,
    -0.0005397928131858655,
    -2.3155014424638595e-05,
    0.001288588538287834,
    0.0023842072601135784,
    0.0015394881792940083,
    0.0010999556380281142,
    0.0010269381790958857,
    0.0012033180907195437,
    0.0010152979330632305,
    0.0012303023771132672,
    0.0017713573348467021,
    0.001651021427551186,
    0.00015196238995195802,
    -0.0006831221915202542,
    -9.172150374333078e-05,
    -0.0009519022687021946,
    -0.0008366177823337817,
    -0.0004617213865976133,
    -0.001555849402924058,
    -0.0012299802881000589,
    0.0005838017378555229,
    0.00040178699964643535,
    0.0002006151411181962,
    -0.0002278699324554417,
    -0.0012630566575317966,
    -0.0007184323441780388,
    0.001050505673054036,
    0.0015754505565434554,
    0.002073173328372481,
    0.0033806478506284436,
    0.0033006972435520796,
]
coarse_basis = PiecewiseLinearBasis(
    np.array([float(key) for key in default_grid.keys()]), width=2.0
)
x0 = coarse_basis.extrapolate([0.0] + list(coarse_space_control) + [0.0])

# x0 = [0.0 for _ in range(len(default_grid))]
top_bound = [0.015 for i in range(len(x0))]
low_bound = [-0.015 for i in range(len(x0))]
# The first and last elements of the control are zero by default
top_bound = solver.linear_basis.discretize(top_bound)[1:-1]
low_bound = solver.linear_basis.discretize(low_bound)[1:-1]
bnds = list(zip(low_bound, top_bound))
# x0 = solver.linear_basis.discretize(x0)[1:-1]
x0 = fine_space_control

run_taylor_test = False
if run_taylor_test:
    energy = []
    state = solver._objective_state(x0)
    energy.append(solver._objective(state))
    print(energy[-1])
    grad = solver._jacobian(state)

    for i in range(1, 11):
        new_state = solver._objective_state(grad * 1.0e-3 * i)
        energy.append(solver._objective(new_state))

    exit()

res = solver.minimize(x0, bnds)
