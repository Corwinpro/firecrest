import decimal
from collections import OrderedDict
import warnings


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

    TODO:
    - recent -> (first, last)
    """

    def __init__(self, state=None, start_time=None):
        decimal.getcontext().prec = 5
        self._dt = None
        self._recent = None

        super().__init__()
        if state:
            self[decimal.Decimal(start_time)] = state
            self._recent = decimal.Decimal(start_time)

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
        return new_instance

    def _same_grid(self, other):
        for el in self:
            if el not in other:
                return False
        return True

    @property
    def recent(self):
        """
        :return: the most recent added state to the State history
        """
        if self._recent is None:
            return None
        return self[self._recent]

    @recent.setter
    def recent(self, value):
        self._recent = value

    def __setitem__(self, key, value):
        key = decimal.Decimal(key)

        try:
            _dt = key - self._recent
        except TypeError:
            pass
        else:
            if self._dt and self._dt != _dt:
                warnings.warn(
                    f"The time series time intervals appear to be non-uniform, with current dt = {self._dt} != _dt",
                    RuntimeWarning,
                )
            self._dt = _dt

        self._recent = key
        super().__setitem__(key, value)

    @classmethod
    def from_dict(cls, dict, reversed=False):
        """
        Create a TimeSeries instance from an (unordered) dict of time stamps

        :param dict: data to TimeSeries
        :param reversed: if the recent element is the first or last in time
        :return: TimeSeries instance
        """
        instance = cls()
        for el in sorted(dict):
            instance[el] = dict[el]
        try:
            instance.recent = min(dict) if reversed else max(dict)
        except ValueError:
            pass
        return instance

    def apply(self, func):
        """
        Applies a function to the TimeSeries instance, returning a new one.

        :param func: a function to apply
        :return: TimeSeries object from the current one
        """
        return self.__class__().from_dict({time: func(self[time]) for time in self})

    def integrate(self, mid_point=True):
        if mid_point:
            return sum(self.values()) * float(self._dt)
        raise NotImplementedError

    @classmethod
    def interpolate_to_keys(cls, series, keys_series):
        if len(series) - 1 != len(keys_series):
            raise TimeGridError(
                "The interpolated series must be of length of keys plus 1"
            )
        instance = cls()
        for key in keys_series:
            instance[key] = 0.5 * (
                series[key - keys_series._dt / decimal.Decimal(2.0)]
                + series[key + keys_series._dt / decimal.Decimal(2.0)]
            )
        return instance
