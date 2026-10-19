from unittest.mock import Mock

import pytest
import torch
import yaml
from pydantic import ValidationError
from yaloader import ConfigLoader, YAMLConfigDumper

from mllooper import State
from mllooper.trainer.trainer import PrecisionAutoCast, PrecisionAutoCastConfig


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
@pytest.mark.parametrize("dtype,expected", [(None, None), ("float16", torch.float16), ("bfloat16", torch.bfloat16)])
def test_precision_autocast_forwards_dtype(monkeypatch, device_type, dtype, expected):
    autocast = Mock()
    monkeypatch.setattr(torch, "autocast", autocast)

    PrecisionAutoCast(device_type=device_type, dtype=dtype)

    autocast.assert_called_once_with(device_type=device_type, dtype=expected)


@pytest.mark.parametrize(
    "option,expected",
    [("", None), ("dtype: null", None), ("dtype: float16", "float16"), ("dtype: bfloat16", "bfloat16")],
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
        device_type="cuda", dtype=getattr(torch, expected) if expected is not None else None
    )


def test_precision_autocast_config_rejects_unsupported_dtype():
    with pytest.raises(ValidationError, match="dtype"):
        PrecisionAutoCastConfig.model_validate({"device_type": "cpu", "dtype": "float32"})


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
def test_precision_autocast_cuda_bfloat16():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    module = PrecisionAutoCast(device_type="cuda", dtype="bfloat16")
    module.initialise({})
    module.initialise({})
    state = State()
    inputs = torch.ones(2, 2, device="cuda")

    module.step(state)
    try:
        assert (inputs @ inputs).dtype == torch.bfloat16
    finally:
        module.step(state)

    assert (inputs @ inputs).dtype == torch.float32
