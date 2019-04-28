from abc import ABC, abstractmethod
import dolfin as dolf

LOG_LEVEL = 30


class BaseSolver(ABC):
    def __init__(self, domain, **kwargs):
        dolf.set_log_level(LOG_LEVEL)
        self.domain = domain

    @abstractmethod
    def solve(self):
        pass
