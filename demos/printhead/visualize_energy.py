from firecrest.misc.time_storage import PiecewiseLinearBasis, TimeSeries
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from decimal import Decimal

rc("text", usetex=True)
rc("font", size=16)

fig = plt.figure(figsize=(6, 8))
gs = fig.add_gridspec(ncols=1, nrows=2)  # , width_ratios=[1, 0.05, 1])

timer = {"dt": Decimal("0.001"), "T": Decimal("2.0")}
default_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)
five_control = [
    0.0016415311773756553,
    0.0002877549134009383,
    0.002814305524765201,
    -0.000577860701974489,
    0.0032713694963792134,
    -0.0006372529768249147,
    0.0021834503389561635,
]
coarse_space_control = [
    0.00311044784454318,
    0.0004090274154494449,
    -0.000885990885988714,
    -0.0008301931565113494,
    0.0006138605251909666,
    0.003079715517723568,
    0.002130128882878145,
    0.00031286018106521223,
    0.0004994208641695934,
    0.00023821240779299953,
    0.0005565306968499129,
    7.294598827090244e-05,
    0.0023074793636292004,
    0.0028340049208040842,
    0.0002856455582957845,
    -0.000994241555623156,
    -0.000504382940868351,
    0.0005828007281855165,
    0.0029586014808884215,
]
fine_space_control = [
    0.006123724356957945,
    0.004864424218755868,
    0.002812486906877589,
    0.0013326145798388984,
    -3.1759925266723935e-05,
    -0.0031763997361657055,
    -0.003540376061713025,
    2.7550037398209843e-05,
    0.0010361022277855265,
    0.0013595264652632468,
    0.0014505649583813763,
    -0.0027341829177681756,
    -0.0023355200433395457,
    -0.0015256442848239893,
    -0.0017572337077866812,
    -0.001540553484138201,
    0.0002005450760386685,
    0.0001649958161787806,
    0.0017145032326317318,
    0.004104251502438635,
    0.004027477467633509,
    0.0019338634856222153,
    0.00012820665456655684,
    -0.0011139630074366276,
    -0.0013021812150029548,
    0.0008856247934691285,
    0.003368367288961495,
    0.0044658677625972224,
    0.002075914630338951,
    -0.0004492783954585621,
    -0.002001578933731112,
    -0.0012611390756716249,
    0.00046346210132873283,
    0.0018677124965585494,
    0.0016548572044693638,
    3.40152283107441e-05,
    -0.00014055244920634126,
    -0.0003207862942244637,
    0.00011584691932131352,
    0.0008754681667647151,
    -0.0001833995938479735,
    -0.0007447724930047289,
    -0.0005638357434997451,
    2.0381541144224558e-05,
    0.0017079018890266793,
    0.00181001784730262,
    0.0005540838584092398,
    -0.001236956214872918,
    -0.001585691370292542,
    7.048090844344718e-05,
    0.002531373080649234,
    0.004161372494684865,
    0.0029336328219073836,
    0.000659848804620981,
    -0.000925510527771468,
    -0.0007301687542686332,
    0.00018411879308577084,
    0.0018239685868184803,
    0.003302092974030934,
    0.0028070527926356476,
    0.0006850293446752663,
    -0.0004980975015519071,
    -0.0002473213351642815,
    -0.001297707653327444,
    -0.0014617996179463531,
    -0.0012312551405898506,
    -0.002064689933263628,
    -0.0012349856230952552,
    0.0013329772920270802,
    0.0014495738050456961,
    0.0009003723476031581,
    -0.0003114524540753525,
    -0.0021724572602572294,
    -0.0018140623460237422,
    0.00023438227585970034,
    0.0015649834251448172,
    0.002885244892033783,
    0.004149001282978173,
    0.0034370137449727723,
]
linear_basis = PiecewiseLinearBasis(
    np.array([float(key) for key in default_grid.keys()]), width=0.05
)
control = [0.0] + fine_space_control + [0.0]
y = linear_basis.extrapolate(control)
x = linear_basis.space
files = [
    "demos/printhead/energy_no_control.dat",
    "demos/printhead/energy_coarse_space_control.dat",
    "demos/printhead/energy_one_control.dat",
    "demos/printhead/energy_fine_control.dat",
]
linestyles = ["-", "-.", ":", "--"]
windows = [r"$\mathrm{No \ control}$", 0.5, 0.1, 0.05]
ax = fig.add_subplot(gs[0, ::])
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
# plt.xlabel(r"$\mathrm{time}, \mu \mathrm{s}$")
plt.ylabel(r"$\epsilon^{-1}\tilde{E}_{\mathrm{free}}$")
ax.set_ylim(-1.0e-5, 3.7e-4)
ax.grid(True)
ax.set_xticklabels([])


for i, file in enumerate(files):
    with open(file) as f:
        data = [x.split() for x in f.read().splitlines()]
        data = [[float(val) for val in line] for line in data]
        ax.plot(
            x[:-1],
            [el[1] for el in data],
            linestyles[i],
            color="k",
            label=(f"$w = {windows[i]}$" if i > 0 else windows[0]),
        )
plt.legend(frameon=True, ncol=2, loc=1)

ax = fig.add_subplot(gs[1, ::])
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_ylim(-1.0e-5, 3.7e-4)
plt.xlabel(r"$\mathrm{time}, \mu \mathrm{s}$")
plt.ylabel(r"$\hat{\mathcal{E}}_{\mathrm{ac}}$")
plt.grid(True)

for i, file in enumerate(files):
    with open(file) as f:
        data = [x.split() for x in f.read().splitlines()]
        data = [[float(val) for val in line] for line in data]
        ax.plot(x[:-1], [el[0] for el in data], linestyles[i], color="k")
# plt.plot(
#     x[:: int((len(y) - 1) / (len(control) - 1))],
#     y[:: int((len(y) - 1) / (len(control) - 1))] / 250,
#     "-o",
#     ms=4,
# )
plt.savefig("energy_history.pdf", bbox_inch="tight")
fig.tight_layout()
plt.show()
