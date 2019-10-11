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

elsize = 0.08
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
    el_size=elsize / 8.0,
    bcond={"noslip": True, "adiabatic": True},
)
boundary2 = LineElement(
    control_points_2,
    el_size=elsize / 8.0,
    bcond={"normal_force": None, "adiabatic": True},
)
boundary3 = LineElement(
    control_points_3, el_size=elsize / 2.0, bcond={"slip": True, "adiabatic": True}
)
boundary4 = LineElement(
    control_points_4,
    el_size=elsize / 20.0,
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
solver = OptimizationSolver(domain, Re=5.0e3, Pr=10.0, timer=timer, signal_window=1.0)
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
    0.0057340410583603,
    0.004348988141468184,
    0.002600713271305053,
    0.0014588122838216863,
    -1.3963252799274768e-05,
    -0.0029536265587314287,
    -0.003208008239608044,
    0.0003867774901876068,
    0.0011280819983280935,
    0.0011591353299198544,
    0.0012764097182818887,
    -0.002460916367586768,
    -0.0020178554770998564,
    -0.0013670691905502443,
    -0.0017718006620248707,
    -0.0014969825370280986,
    0.00018035963910830908,
    6.486019238675759e-05,
    0.0012411480058023231,
    0.0038396199078573755,
    0.0037676162665253345,
    0.001823109793081534,
    0.00011139038027340335,
    -0.0010031191899376262,
    -0.00108534994537646,
    0.0010568937508934734,
    0.0033569215955585575,
    0.004203841393821547,
    0.002000569397421494,
    -0.00021343583507798738,
    -0.0017699617373726705,
    -0.000972946515541353,
    0.0005112773977696465,
    0.0017117389239682498,
    0.0015319530970557278,
    -7.435667663405189e-05,
    -0.00033002526087062253,
    -0.00047180156257920365,
    0.00011596815398274645,
    0.0010414846809007063,
    5.188267208057239e-05,
    -0.0008583016120634149,
    -0.0005342174190514572,
    7.876803304607066e-05,
    0.0015837498050353459,
    0.0015961162466011705,
    0.0005303669690068481,
    -0.0011343499844002874,
    -0.0014369592188818237,
    0.0002861077392815923,
    0.0022854020225331205,
    0.003921745818550618,
    0.002976291541322398,
    0.0007062540246278567,
    -0.000696818289551337,
    -0.0007018867800509128,
    7.956070992902062e-05,
    0.0017142656681538588,
    0.0031626368654953437,
    0.002571209478925508,
    0.0006842610657258326,
    -0.00041524485108104483,
    -0.00010103016510627052,
    -0.001098398530256575,
    -0.0014336849594044677,
    -0.0013005192315957358,
    -0.0018954428885900303,
    -0.0011654245455391394,
    0.0010711926003203391,
    0.0013096432958956883,
    0.0009142349409793696,
    -0.00028351955196745686,
    -0.0018094588580020877,
    -0.001575808500947884,
    0.00015534184364729588,
    0.0014956082687823632,
    0.002817621410850364,
    0.003857146725632634,
    0.0032568677076993194,
]
one_space_control = [
    0.005324010468649279,
    0.003343322647967684,
    0.00016064736472172868,
    -0.002609956815594884,
    0.00016017495056383385,
    -0.0017920392639235205,
    -0.0011684353053945725,
    -0.0011640435333961354,
    -0.0003478163580602155,
    0.002680230469332851,
    0.0035283592996153594,
    0.0012350678474811394,
    0.0020440647579950525,
    0.0030601016469968645,
    -0.001300349273811132,
    -0.0013562631408648453,
    0.002429822124366182,
    0.0011903783250840553,
    -0.0004978743810532501,
    0.00012618190506874136,
    -0.0007177418693306068,
    0.0008351849958615962,
    0.0023076761486244936,
    -0.0008946019301380705,
    -0.0006792350052757791,
    0.0031896064520691297,
    0.0019994318287890004,
    0.0011238524318151117,
    0.0026527315266341666,
    0.0017623738953415598,
    -0.0011358134563953682,
    -0.0010213713405194979,
    -0.000925244092293188,
    -0.0008996721045075374,
    0.00044937109726455893,
    -0.0014161497395480105,
    0.00023472697596130617,
    0.0030337916492754143,
    0.0039864338178979535,
]
# coarse_basis = PiecewiseLinearBasis(
#     np.array([float(key) for key in default_grid.keys()]), width=2.0
# )
# x0 = coarse_basis.extrapolate([0.0] + list(coarse_space_control) + [0.0])

x0 = [0.0 for _ in range(len(default_grid))]
top_bound = [0.015 for i in range(len(x0))]
low_bound = [-0.015 for i in range(len(x0))]
# The first and last elements of the control are zero by default
top_bound = solver.linear_basis.discretize(top_bound)[1:-1]
low_bound = solver.linear_basis.discretize(low_bound)[1:-1]
bnds = list(zip(low_bound, top_bound))
x0 = solver.linear_basis.discretize(x0)[1:-1]
x0 = one_space_control

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
