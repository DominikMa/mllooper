import logging
from dataclasses import dataclass

logger = logging.getLogger('State')

@dataclass
class State:
    pass

    def __getattr__(self, name):
        raise AttributeError(
            f"The current state has no attribute {name}. "
            f"It seems that some module relies on {name}, but it is missing."
        )