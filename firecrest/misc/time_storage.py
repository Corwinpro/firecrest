import decimal
from collections import OrderedDict


class StateHistory(OrderedDict):
    """
    State storage class for saving state snapshots at different time steps.
    """

    def __init__(self, state=None, start_time=None):
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

    def __setitem__(self, key, value):
        self._recent = decimal.Decimal(key)
        super().__setitem__(decimal.Decimal(key), value)
