import logging
from abc import ABC
from typing import Optional

from yaloader import YAMLBaseConfig, loads

from mllooper import State


class StateTest(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(self.name)

    def __call__(self, state: State) -> bool:
        raise NotImplementedError


@loads(None)
class StateTestConfig(YAMLBaseConfig, ABC):
    name: Optional[str] = None

