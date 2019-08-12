import decimal
from collections import OrderedDict


class TimeSeries(OrderedDict):
    """
    State storage class for saving state snapshots at different time steps.
    """

    def __init__(self, state=None, start_time=None):
        decimal.getcontext().prec = 5
        super().__init__()
        if state:
            self[decimal.Decimal(start_time)] = state
            self._recent = decimal.Decimal(start_time)

    @property
    def recent(self):
        """
        :return: the most recent added state to the State history
        """
        return self[self._recent]

    @recent.setter
    def recent(self, value):
        self._recent = value

    def __setitem__(self, key, value):
        key = decimal.Decimal(round(key, 5))
        self._recent = key
        super().__setitem__(key, value)

    @classmethod
    def from_dict(cls, dict, reversed=False):
        instance = cls()
        for el in dict:
            instance[el] = dict[el]
        instance.recent = min(dict) if reversed else max(dict)
        return instance

    def apply(self, func):
        d = {time: func(self[time]) for time in self}
        return TimeSeries.from_dict(d)
