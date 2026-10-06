from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

from yaloader import YAMLBaseConfig, loads

if TYPE_CHECKING:
    from mllooper import State


class StateTest(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def __call__(self, state: State) -> bool:
        raise NotImplementedError


@loads(None)
class StateTestConfig(YAMLBaseConfig, ABC):
    name: str | None = None
