from firecrest.misc.time_storage import PiecewiseLinearBasis, TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal


timer = {"dt": Decimal("0.01"), "T": Decimal("1.2")}
default_grid = TimeSeries.from_dict(
    {
        Decimal(k) * Decimal(timer["dt"]): (k * float(timer["dt"])) ** 2
        for k in range(int(timer["T"] / Decimal(timer["dt"])) + 1)
    }
)

x = np.array([float(key) for key in default_grid.keys()])
y = np.array([val for val in default_grid.values()])

basis = PiecewiseLinearBasis(x, 0.25)
plt.plot(x, y)
plt.plot(x, basis.project(y))
plt.show()
