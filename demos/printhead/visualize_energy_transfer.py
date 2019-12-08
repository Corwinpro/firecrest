from firecrest.misc.time_storage import PiecewiseLinearBasis, TimeSeries
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from decimal import Decimal
import scipy.stats as stats

rc("text", usetex=True)
rc("font", size=16)

fig = plt.figure(figsize=(15, 4))

timer = {"dt": Decimal("0.001"), "T": Decimal("2.0")}
default_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)
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

linear_basis = PiecewiseLinearBasis(
    np.array([float(key) for key in default_grid.keys()]), width=0.05
)
control = [0.0] + fine_space_control + [0.0]
y = linear_basis.extrapolate(control)
x = linear_basis.space

ax = fig.add_subplot(111)
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_ylim(-1.2e-2, 2.0e-2)
ax.plot(
    x[:: int((len(y) - 1) / (len(control) - 1))],
    y[:: int((len(y) - 1) / (len(control) - 1))] / 10 ** 0.5,
    "-o",
    ms=4,
    color="k",
    alpha=0.8,
)
ax.vlines(
    x[:: int((len(y) - 1) / (len(control) - 1))],
    ymin=-1.5e-2,
    ymax=3.0e-2,
    linewidth=0.075,
    alpha=0.3,
)
plt.grid(True)
plt.ylabel(r"$\mathcal{U}(t)$", fontsize=22)
plt.xlabel(r"$\mathrm{time}, \mu \mathrm{s}$")

file = "demos/printhead/energy_fine_control.dat"
# file = "demos/printhead/energy_one_control.dat"

ax2 = ax.twinx()
ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2.set_ylim(-6.0e-4, 1.0e-3)
ax2.tick_params(axis="y", colors="red")

with open(file) as f:
    data = [x.split() for x in f.read().splitlines()]
    data = np.array([[float(val) for val in line] for line in data])
    # ax2.plot(
    #     x[1:-1],
    #     ((data[1:, 0] - data[:-1, 0]) + (data[1:, 1] - data[:-1, 1])) / 0.01,
    #     color="k",
    # )
    # ax2.plot(x[1:-1], ((data[1:, 0] - data[:-1, 0])) / 0.001, color="r")
    ax2.plot(x[:-1], data[:, 2], color="r")

    r, p = stats.pearsonr(data[:, 2], -data[::-1, 2])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    r, _ = stats.pearsonr(data[:1000, 2], y[:-1001])
    print(f"Scipy computed Pearson r: {r}")
    r, _ = stats.pearsonr(data[1000:, 2], y[1001:])
    print(f"Scipy computed Pearson r: {r}")

plt.ylabel(r"$\hat{\mathcal{F}}_{\Gamma_{\mathrm{opt}}}$", color="r", fontsize=22)

fig.tight_layout()
plt.savefig("fine_energy_transfer.pdf", bbox_inch="tight")
plt.show()
