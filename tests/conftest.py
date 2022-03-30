import pytest

from mllooper import State


@pytest.fixture
def empty_state():
    return State()
