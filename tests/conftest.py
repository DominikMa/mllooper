import pytest
from baselooper import State


@pytest.fixture
def empty_state():
    return State()
