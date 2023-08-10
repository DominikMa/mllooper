import pydantic
import pytest

from mllooper import LooperConfig, NOPConfig, LooperIterationStopConfig, LooperIterationStop, Looper


def test_extra_key_module_in_modules():
    config = LooperConfig(
        modules={
            'module': NOPConfig()
        },
        another_module=NOPConfig()
    )
    looper = config.load()
    assert 'another_module' in looper.modules


def test_extra_key_reference_in_modules():
    config = LooperConfig(
        modules={
            'module': NOPConfig()
        },
        another_module='module'
    )
    looper = config.load()
    assert 'another_module' in looper.modules


def test_extra_key_already_in_modules():
    with pytest.raises(pydantic.ValidationError):
        LooperConfig(
            modules={
                'another_module': NOPConfig(),
            },
            another_module='module'
        )


def test_missing_reference():
    with pytest.raises(pydantic.ValidationError):
        LooperConfig(
            modules={
                'module': NOPConfig(),
                'ref_name': 'ref'
            }
        )

def test_initialise_and_teardown_is_called(initialise_and_teardown_counter_class):
    counter = initialise_and_teardown_counter_class()
    looper = Looper(
        modules={
            'module': counter,
            'stop': LooperIterationStop(step_iteration_limit=1)
        }
    )
    looper.run()

    assert counter.count_initialise == 1
    assert counter.count_teardown == 1


def test_initialise_and_teardown_for_references_is_called(initialise_and_teardown_counter_class):
    counter = initialise_and_teardown_counter_class()
    looper = Looper(
        modules={
            'module': counter,
            'module2': 'module',
            'another_module': 'module',
            'stop': LooperIterationStop(step_iteration_limit=1)
        }
    )
    looper.run()

    assert counter.count_initialise == 3
    assert counter.count_teardown == 3


def test_initialise_and_teardown_for_references_is_called_config(initialise_and_teardown_counter_class_config):
    config = LooperConfig(
        modules={
            'module': initialise_and_teardown_counter_class_config(),
            'module2': 'module',
        },
        another_module='module',
        another_module2='module',
        stop=LooperIterationStopConfig(step_iteration_limit=1)
    )
    looper = config.load()
    counter = looper.modules['module']

    looper.run()

    assert counter.count_initialise == 4
    assert counter.count_teardown == 4
