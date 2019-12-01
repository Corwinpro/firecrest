from firecrest.misc.time_storage import PiecewiseLinearBasis, TimeSeries
import matplotlib.pyplot as plt
from matplotlib import rc, gridspec
import numpy as np
from decimal import Decimal

rc("text", usetex=True)
rc("font", size=16)

fig = plt.figure(figsize=(12, 6))
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
linear_basis = PiecewiseLinearBasis(
    np.array([float(key) for key in default_grid.keys()]), width=0.2
)
control = [0.0] + coarse_space_control + [0.0]
y = linear_basis.extrapolate(control)
x = linear_basis.space
plt.plot(
    x[:: int((len(y) - 1) / (len(control) - 1))],
    y[:: int((len(y) - 1) / (len(control) - 1))],
    "-o",
    ms=4,
)
with open("demos/printhead/energy_coarse_space_control.dat") as f:
    data = [x.split() for x in f.read().splitlines()]
    data = [[float(val) for val in line] for line in data]
    plt.plot(x[:-1], [el[1] * 100 for el in data])
    plt.plot(
        x[:-2], [50000 * (data[i + 1][1] - data[i][1]) for i in range(len(data) - 1)]
    )
    plt.plot(x[:-1], [el[0] * 100 for el in data])
    plt.plot(
        x[:-2], [50000 * (data[i + 1][0] - data[i][0]) for i in range(len(data) - 1)]
    )

# with open("demos/printhead/energy_no_control.dat") as f:
#     data = [x.split() for x in f.read().splitlines()][:2000]
#     data = [[float(val) for val in line] for line in data]
#     # plt.plot(x[:-1], [el[1] * 100 for el in data])
#     plt.plot(x[:-2], [data[i + 1][0] - data[i][0] for i in range(len(data) - 1)])


plt.show()
