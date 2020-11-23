import decimal
from decimal import Decimal
from collections import OrderedDict
import warnings
import numpy as np


class TimeGridError(Exception):
    def __init__(self, message=None):
        if not message:
            message = (
                "The multiplication expects the objects to have compatible time grids"
            )
        super().__init__(message)


class TimeSeries(OrderedDict):
    """
    State storage class for saving state snapshots at different time steps.
    """

    def __init__(self, state=None, start_time=None):
        self._dt = None
        self._first = None
        self._last = None

        super().__init__()
        if state:
            start_time = decimal.Decimal(start_time)
            self[start_time] = state
            self._first = start_time
            self._last = start_time

    def __mul__(self, other):
        if not self._same_grid(other):
            interpolated = self.__class__.interpolate_to_keys(other, self)
            if not self._same_grid(interpolated):
                raise TimeGridError
            else:
                other = interpolated

        new_instance = self.__class__()
        for el in self:
            new_instance[el] = self[el] * other[el]

        if len(new_instance) == 1:
            new_instance._dt = self._dt or other._dt
        return new_instance

    def _same_grid(self, other):
        for el in self:
            if el not in other:
                return False
        return True

    @property
    def first(self):
        if self._first is None:
            return None
        return self[self._first]

    @property
    def last(self):
        if self._last is None:
            return None
        return self[self._last]

    def _warn_time_interval_consistency(self, key):
        """
        Check if adding a new key changes the time step.
        :param key: Key to insert
        :return: None
        """
        try:
            _dt = min(abs(key - self._first), abs(key - self._last))
        except TypeError:
            pass
        else:
            if self._dt and self._dt != _dt:
                warnings.warn(
                    f"The time series time intervals appear to be non-uniform, "
                    f"with current dt = {self._dt} != {_dt}",
                    RuntimeWarning,
                )
            self._dt = _dt

    def __setitem__(self, key, value):
        key = decimal.Decimal(key)
        if key not in self:
            self._warn_time_interval_consistency(key)

        self._first = min(i for i in (key, self._first) if i is not None)
        self._last = max(i for i in (key, self._last) if i is not None)
        super().__setitem__(key, value)

    def values(self):
        return [el[1] for el in sorted(self.items())]

    @classmethod
    def from_dict(cls, dict):
        """
        Create a TimeSeries instance from an (unordered) dict of time stamps

        :param dict: data to TimeSeries
        :return: TimeSeries instance
        """
        instance = cls()
        if len(dict) == 0:
            return instance
        for el in sorted(dict):
            instance[el] = dict[el]

        return instance

    @classmethod
    def from_list(cls, array, template_grid):
        """
        Create a TimeSeries instance from a list and a template TimeSeries (grid)

        :param array: data to TimeSeries
        :param template_grid: template TimeSeries for data storage
        :return: TimeSeries instance
        """
        if len(array) != len(template_grid):
            raise TimeGridError(
                f"Array len {len(array)} must be of the same size "
                f"as template grid ({len(template_grid)})"
            )
        instance = cls()
        for item, key in zip(array, template_grid):
            instance[key] = item

        return instance

    def _from_list(self, array):
        return self.__class__.from_list(array, self)

    @classmethod
    def from_parameters(cls, start, stop, step):
        if not isinstance(start, Decimal):
            start = Decimal(start)
        if not isinstance(stop, Decimal):
            stop = Decimal(stop)
        if not isinstance(step, Decimal):
            step = Decimal(step)
        instance = TimeSeries.from_dict(
            {
                start + Decimal(k) * step: 0
                for k in range(int((stop - start) / step) + 1)
            }
        )
        return instance

    def apply(self, func):
        """
        Applies a function to the TimeSeries instance, returning a new one.

        :param func: a function to apply
        :return: TimeSeries object from the current one
        """
        return self.__class__().from_dict({time: func(self[time]) for time in self})

    def integrate(self, mid_point=True):
        result = sum(self.values()) * float(self._dt)
        if mid_point:
            return result
        return result - (self.first + self.last) * float(self._dt) / 2.0

    @classmethod
    def interpolate_to_keys(cls, series, keys_series):
        """
        Given the `series` TimeSeries object, we interpolate the `series` values
        onto the `keys_series` TimeSeries keys.
        :param series:TimeSeries with values
        :param keys_series:TimeSeries with keys
        :return:TimeSeries
        """
        _copy_series = TimeSeries.from_dict(series)

        if len(_copy_series) == len(keys_series) - 1:
            _copy_series[_copy_series._first - _copy_series._dt] = -_copy_series[
                _copy_series._first
            ]
            _copy_series[_copy_series._last + _copy_series._dt] = -_copy_series[
                _copy_series._last
            ]

        if len(_copy_series) - 1 != len(keys_series):
            raise TimeGridError(
                "The interpolated series must be of length of keys plus 1"
            )
        instance = cls()
        dt = keys_series._dt
        if dt is None:
            dt = _copy_series._dt
            warnings.warn(
                f"The interpolation TimeSeries grid has no dt attribute. "
                f"Using the interpolating grid dt={dt} instead"
            )

        for key in keys_series:
            instance[key] = 0.5 * (
                _copy_series[key - dt / decimal.Decimal(2.0)]
                + _copy_series[key + dt / decimal.Decimal(2.0)]
            )

        if len(instance) == 1:
            instance._dt = _copy_series._dt or keys_series._dt
        return instance

    def interpolate(self, series):
        return self.__class__.interpolate_to_keys(series, self)


class PiecewiseLinearBasis:
    """
    Creates a triangular Piecewise Linear Basis for the given `space`.
    The width of the triangles is defined by the `width` parameter.

    :kw param reduced_basis: sets the basis space to zero at corners (H_0)
    """

    def __init__(self, space, width, **kwargs):
        self.width = width
        self.space = space
        self.space_step_size = abs(self.space[1] - self.space[0])

        assert (self.space[-1] - self.space[0]) % self.width < 1.0e-8 or (
            self.space[-1] - self.space[0]
        ) % self.width > self.width - 1.0e-8, (
            f"Width {self.width} must be compatible"
            + f"with the domain length ({self.space[-1] - self.space[0]})"
        )
        self.basis = []
        self._is_reduced_basis = kwargs.get("reduced_basis", False)
        self._construct_basis()

        self.mass_matrix = None
        self.inv_mass_matrix = None
        self._construct_mass_matrix()

    def _basis_function(self, mid, width=None, space=None):
        """
        Given a mid point, an interval width, and a discrete space, we construct a piecewise linear
        basis function on it.
        :param mid:float mid point for basis function
        :param width:float basis function width
        :param space:np.array discrete space
        :return:np.array basis function array
        """
        if width is None:
            width = self.width
        if space is None:
            space = self.space
        # (Optional) Normalization constant for basis function
        h = (3.0 / 2.0 / (width / 2.0)) ** 0.5

        # Placeholder for the basis function
        y = space * 0

        for i in range(len(space)):
            if (space[i] >= mid - (width / 2.0)) and (space[i] <= mid):
                y[i] = (space[i] - mid + (width / 2.0)) / (width / 2.0) * h
            elif (space[i] <= mid + (width / 2.0)) and (space[i] >= mid):
                y[i] = h - (space[i] - mid) / (width / 2.0) * h
        return y

    def _construct_basis(self):
        if self._is_reduced_basis:
            position_offsets = self.width / 2.0
        else:
            position_offsets = 0.0
        position = self.space[0] + position_offsets
        while abs(position - self.space[-1]) > 1.0e-12:
            self.basis.append(self._basis_function(position))
            position += self.width / 2.0
        if not self._is_reduced_basis:
            self.basis.append(self._basis_function(self.space[-1]))

    def _construct_mass_matrix(self):
        self.mass_matrix = np.array(
            [
                [sum(b1 * b2) * self.space_step_size for b1 in self.basis]
                for b2 in self.basis
            ]
        )
        self.inv_mass_matrix = np.linalg.inv(self.mass_matrix)

    def project(self, y):
        """
        Projects the y-function onto the discrete piecewise linear space
        :param y:np.array function to project
        :return:np.array projected form of the y-function
        """
        coefficients = self.discretize(y)
        return self.extrapolate(coefficients)

    def discretize(self, y, *, mid_point=False):
        """
        Calculates a discrete lower-order space representation of the y-function
        :param y:np.array function to discretize
        :param mid_point:bool grid or mid points provided
        :return:list discrete coefficients
        """
        if mid_point:
            signal = (
                [0.0] + [0.5 * (y[i] + y[i + 1]) for i in range(len(y) - 1)] + [0.0]
            )
        else:
            signal = y
        return self.inv_mass_matrix.dot(
            np.array([sum(signal * b) * self.space_step_size for b in self.basis])
        )

    def extrapolate(self, coefficients):
        """
        Restores the linear function on self.space from discrete coefficients
        :param coefficients:np.array discrete space coefficients
        :return:np.array restored function
        """
        if len(coefficients) != len(self.basis):
            raise ValueError(
                f"Number of coefficients {len(coefficients)} should equal the basis length {len(self.basis)}"
            )
        res = self.space * 0
        for i in range(len(self.basis)):
            res += coefficients[i] * self.basis[i]

        return res

    @classmethod
    def convert_between_spaces(cls, from_basis, data, to_width):
        """ Given the `data` array representation of a signal on
        the `from_basis`, converts this discrete signal (reducing or
        increasing the number of discrete points) to a signal with `to_width`
        discrete basis width

        :return
        new_data: array
            representation of the `data` on the new space of width `to_width`
        new_bases: PiecewiseLinearBasis
            basis with `to_width` width
        """
        from_signal = from_basis.extrapolate(data)

        space = from_basis.space
        new_basis = cls(space, to_width, reduced_basis=from_basis._is_reduced_basis)
        return new_basis.discretize(from_signal), new_basis


class ControlSpaceFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def create_piecewise_linear_space(self):
        return LinearControlSpace(*self.args, **self.kwargs)


class LinearControlSpace:
    def __init__(self, dt, final_time, signal_window, *args, **kwargs):
        final_time = Decimal(final_time)

        # 1. define control
        # `time_grid` defines the time grid for the direct simulation
        self.time_grid = TimeSeries.from_dict(
            {
                Decimal(k) * Decimal(dt): 0
                for k in range(int(final_time / Decimal(dt)) + 1)
            }
        )
        # `midpoint_grid` defines the time grid for the adjoint simulation
        self.midpoint_grid = TimeSeries.from_dict(
            {
                Decimal(k + 0.5) * Decimal(dt): 0
                for k in range(int(final_time / Decimal(dt)))
            }
        )
        # We define a control space, a PiecewiseLinearBasis in this case
        self._is_reduced_basis = kwargs.get("reduced_basis", True)
        self.basis = PiecewiseLinearBasis(
            np.array([float(key) for key in self.time_grid.keys()]),
            width=signal_window,
            reduced_basis=self._is_reduced_basis,
        )

    def discrete_to_continuous(self, discrete_values):
        continuous_values = self.basis.extrapolate(list(discrete_values))
        return continuous_values

    def continuous_to_discrete(self, function):
        return self.basis.discretize(function.values())
