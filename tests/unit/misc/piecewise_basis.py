from firecrest.misc.time_storage import PiecewiseLinearBasis, TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal


timer = {"dt": Decimal("0.01"), "T": Decimal("1.2")}
func = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): 10
        + 40.0
        * (k * float(timer["dt"]))
        * (k * float(timer["dt"]) - 0.5)
        * (k * float(timer["dt"]) - 1)
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)

x = np.array([float(key) for key in func.keys()])
y = np.array([val for val in func.values()])

basis = PiecewiseLinearBasis(x, 0.4)
plt.plot(x, y, "k")
plt.plot(x, basis.project(y), "--", color="k")
for b in basis.basis:
    plt.plot(x, b, "-.", color="gray")

plt.grid(True)
plt.show()
