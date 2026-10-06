from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar
from yaloader import YAMLBaseConfig, loads

if TYPE_CHECKING:
    from mllooper import State


class StateTest(ABC):  # noqa: B024 - Base predicate raises until overridden.
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def __call__(self, state: State) -> bool:
        raise NotImplementedError


_StateTest = TypeVar("_StateTest", bound=StateTest, default=Any)


@loads(None)
class StateTestConfig(YAMLBaseConfig[_StateTest], ABC):
    name: str | None = None
