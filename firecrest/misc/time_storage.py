import decimal
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
                    f"The time series time intervals appear to be non-uniform, with current dt = {self._dt} != {_dt}",
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
                f"Array must be of the same size as template grid ({len(template_grid)})"
            )
        instance = cls()
        for item, key in zip(array, template_grid):
            instance[key] = item

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
        if len(series) - 1 != len(keys_series):
            raise TimeGridError(
                "The interpolated series must be of length of keys plus 1"
            )
        instance = cls()
        dt = keys_series._dt
        if dt is None:
            dt = series._dt
            warnings.warn(
                f"The interpolation TimeSeries grid has no dt attribute. Using the interpolating grid dt={dt} instead"
            )

        for key in keys_series:
            instance[key] = 0.5 * (
                series[key - dt / decimal.Decimal(2.0)]
                + series[key + dt / decimal.Decimal(2.0)]
            )

        if len(instance) == 1:
            instance._dt = series._dt or keys_series._dt
        return instance


class PiecewiseLinearBasis:
    """
    Creates a triangular Piecewise Linear Basis for the given `space`.
    The width of the triangles is defined by the `width` parameter.

    :kw param reduced_basis: sets the basis space to zero at corners (H_0)
    """

    def __init__(self, space, width, **kwargs):
        self.space = space
        self.space_step_size = abs(self.space[1] - self.space[0])

        self.width = width
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

    def discretize(self, y):
        """
        Calculates a discrete lower-order space representation of the y-function
        :param y:np.array function to discretize
        :return:list discrete coefficients
        """
        return self.inv_mass_matrix.dot(
            np.array([sum(y * b) * self.space_step_size for b in self.basis])
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
