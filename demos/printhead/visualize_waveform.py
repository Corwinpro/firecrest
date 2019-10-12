from firecrest.misc.time_storage import PiecewiseLinearBasis, TimeSeries
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from decimal import Decimal

timer = {"dt": Decimal("0.01"), "T": Decimal("20.0")}
default_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 0
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)
five_control = [
    0.0012682707839538011,
    0.0011420536132507367,
    0.0011953767253928369,
    0.0013219451301906874,
    0.0011990719615676555,
    0.0008732554807402981,
    0.0013210959159414273,
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


final_energies = [4.7449879358677454e-05, 3.718615040954647e-05, 3.1202885561657123e-05]

controls = [five_control, coarse_space_control, one_space_control, fine_space_control]
windows = (5.0, 2.0, 1.0, 0.5)
colors = ["blue", "black", "gray", "silver"]

for i in range(len(controls)):
    linear_basis = PiecewiseLinearBasis(
        np.array([float(key) for key in default_grid.keys()]), width=windows[i]
    )
    control = [0.0] + controls[i] + [0.0]
    y = linear_basis.extrapolate(control)
    x = linear_basis.space
    plt.plot(x, y, "-", color=colors[i])

plt.show()
