from unittest.mock import Mock

import pytest
import torch
import yaml
from yaloader import ConfigLoader, YAMLConfigDumper

from mllooper import State
from mllooper.trainer.trainer import PrecisionAutoCast, PrecisionAutoCastConfig


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (None, None),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("int64", torch.int64),
        ("complex64", torch.complex64),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_precision_autocast_forwards_dtype(monkeypatch, device_type, dtype, expected):
    autocast = Mock()
    monkeypatch.setattr(torch, "autocast", autocast)

    PrecisionAutoCast(device_type=device_type, dtype=dtype)

    autocast.assert_called_once_with(device_type=device_type, dtype=expected, enabled=True, cache_enabled=None)


@pytest.mark.parametrize(
    "option,expected",
    [
        ("", None),
        ("dtype: null", None),
        ("dtype: float16", "float16"),
        ("dtype: bfloat16", "bfloat16"),
        ("dtype: float32", "float32"),
        ("dtype: float64", "float64"),
        ("dtype: int64", "int64"),
        ("dtype: complex64", "complex64"),
        ("dtype: float", "float"),
    ],
)
def test_precision_autocast_yaml_round_trip(monkeypatch, option, expected):
    loader = ConfigLoader()
    config = loader.construct_from_string(f"!PrecisionAutoCast\ndevice_type: cuda\n{option}\n")
    restored = loader.construct_from_string(yaml.dump(config, Dumper=YAMLConfigDumper))

    assert isinstance(restored, PrecisionAutoCastConfig)
    assert restored.dtype == expected

    autocast = Mock()
    monkeypatch.setattr(torch, "autocast", autocast)
    module = restored.load()

    assert isinstance(module, PrecisionAutoCast)
    autocast.assert_called_once_with(
        device_type="cuda",
        dtype=getattr(torch, expected) if expected is not None else None,
        enabled=True,
        cache_enabled=None,
    )


@pytest.mark.parametrize("dtype", ["not_a_dtype", "Tensor", "autocast"])
def test_precision_autocast_rejects_unknown_dtype_names(dtype):
    config = PrecisionAutoCastConfig(device_type="cpu", dtype=dtype)
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        config.load()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("cache_enabled", [None, True, False])
def test_precision_autocast_options_yaml_round_trip(monkeypatch, enabled, cache_enabled):
    config = PrecisionAutoCastConfig(device_type="cuda", dtype="bfloat16", enabled=enabled, cache_enabled=cache_enabled)
    restored = ConfigLoader().construct_from_string(yaml.dump(config, Dumper=YAMLConfigDumper))
    autocast = Mock()
    monkeypatch.setattr(torch, "autocast", autocast)

    restored.load()

    autocast.assert_called_once_with(
        device_type="cuda", dtype=torch.bfloat16, enabled=enabled, cache_enabled=cache_enabled
    )


@pytest.mark.parametrize("dtype", [None, "float32", "bfloat16"])
@pytest.mark.parametrize("cache_enabled", [None, True, False])
def test_precision_autocast_disabled_restores_outer_context(dtype, cache_enabled):
    default_cache_enabled = torch.is_autocast_cache_enabled()
    module = PrecisionAutoCastConfig(device_type="cpu", dtype=dtype, enabled=False, cache_enabled=cache_enabled).load()
    module.initialise({})
    module.initialise({})
    state = State()
    inputs = torch.ones(2, 2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, cache_enabled=False):
        assert (inputs @ inputs).dtype == torch.bfloat16
        for _ in range(2):
            module.step(state)
            try:
                assert (inputs @ inputs).dtype == torch.float32
                expected_cache_enabled = default_cache_enabled if cache_enabled is None else cache_enabled
                assert torch.is_autocast_cache_enabled() is expected_cache_enabled
            finally:
                module.step(state)

            assert (inputs @ inputs).dtype == torch.bfloat16
            assert torch.is_autocast_cache_enabled() is False
            module.step_callback(state)


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
def test_precision_autocast_cpu_forward_and_backward(dtype):
    module = PrecisionAutoCast(device_type="cpu", dtype=dtype)
    module.initialise({})
    module.initialise({})
    state = State()
    inputs = torch.ones(2, 2, requires_grad=True)

    for _ in range(2):
        module.step(state)
        try:
            output = inputs @ inputs
            assert output.dtype == torch.bfloat16
        finally:
            module.step(state)

        assert (inputs @ inputs).dtype == torch.float32
        output.float().sum().backward()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        inputs.grad = None
        module.step_callback(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("enabled", [True, False])
def test_precision_autocast_cuda_bfloat16(enabled):
    if enabled and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    module = PrecisionAutoCast(device_type="cuda", dtype="bfloat16", enabled=enabled)
    module.initialise({})
    module.initialise({})
    state = State()
    inputs = torch.ones(2, 2, device="cuda")

    module.step(state)
    try:
        assert (inputs @ inputs).dtype == (torch.bfloat16 if enabled else torch.float32)
    finally:
        module.step(state)

    assert (inputs @ inputs).dtype == torch.float32
