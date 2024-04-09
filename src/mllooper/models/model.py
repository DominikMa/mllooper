import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Literal

import torch
import torch.distributed as distributed
from torch import nn
from yaloader import loads

from mllooper import SeededModule, State, SeededModuleConfig, Module, ModuleConfig
from mllooper.data import DatasetState


@dataclass
class ModelState(State):
    output: Optional[Any] = None


class Model(SeededModule, ABC):
    def __init__(
        self,
        torch_model: nn.Module,
        module_load_file: Optional[Path] = None,
        device: Union[str, List[str]] = "cpu",
        output_device: Optional[str] = None,
        state_name_dataset: str = "dataset_state",
        state_name_model: str = "model_state",
        force_gradient: Optional[bool] = None,
        compile_model: bool = False,
        data_parallel: Optional[Literal["DP", "DDP"]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        devices = device if isinstance(device, list) else [device]
        self.devices = [torch.device(device) for device in devices]
        self.device = self.devices[0]
        self.output_device = (
            torch.device(output_device) if output_device is not None else None
        )
        self.module = torch_model.to(self.device)

        self.compile_model = compile_model
        self.data_parallel = data_parallel

        if data_parallel is None:
            self._parallel_module = None
            if len(devices) > 1:
                raise RuntimeError(
                    f"To use multiple devices for the model data_parallel needs to be set to DP or DDP."
                )
        elif data_parallel == "DP":
            self._parallel_module = nn.DataParallel(
                self.module, device_ids=self.devices, output_device=output_device
            )
        elif data_parallel == "DDP":
            self._parallel_module = nn.parallel.DistributedDataParallel(
                self.module, device_ids=self.devices, output_device=output_device
            )
        else:
            raise RuntimeError(f"Unsupported data parallel mode: {data_parallel}")

        self._compiled_module: Optional[nn.Module] = None
        if self.compile_model:
            if self._parallel_module is not None:
                self._compiled_module = torch.compile(self._parallel_module)
            else:
                self._compiled_module = torch.compile(self.module)

        self.state_name_dataset: str = state_name_dataset
        self.state_name_model: str = state_name_model

        if module_load_file:
            module_state_dict = torch.load(module_load_file, map_location=self.device)
            self.module.load_state_dict(module_state_dict)

        self.force_gradient: Optional[bool] = force_gradient
        self.state = ModelState()

    def step(self, state: State) -> None:
        dataset_state: DatasetState = getattr(state, self.state_name_dataset)
        self.state.output = None

        module_input = self.format_module_input(dataset_state.data)

        self.module.train() if dataset_state.train else self.module.eval()
        if self._parallel_module is not None:
            self._parallel_module.train() if dataset_state.train else self._parallel_module.eval()
        if self._compiled_module is not None:
            self._compiled_module.train() if dataset_state.train else self._compiled_module.eval()

        self._parallel_module: Optional[nn.Module]

        with torch.set_grad_enabled(
            self.force_gradient
            if self.force_gradient is not None
            else dataset_state.train
        ):
            if self._compiled_module is not None:
                module_output = self._compiled_module(module_input)
            elif self._parallel_module is not None:
                module_output = self._parallel_module(module_input)
            else:
                module_output = self.module(module_input)

            if self.output_device is not None:
                module_output = module_output.to(self.output_device)

        self.state.output = self.format_module_output(module_output)
        setattr(state, self.state_name_model, self.state)

    @staticmethod
    def format_module_input(data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data
        elif (
            isinstance(data, Dict)
            and "input" in data.keys()
            and isinstance(data["input"], torch.Tensor)
        ):
            return data["input"]
        else:
            raise NotImplementedError

    @staticmethod
    def format_module_output(output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output
        else:
            raise NotImplementedError

    def trainable_parameters(self, param_groups: Optional[List[Dict]]) -> List[Dict]:
        if param_groups is None:
            return [{"params": self.module.parameters()}]
        # Make a copy so that the parameters do not end up in the pydantic model
        param_groups = [d.copy() for d in param_groups]
        if len(param_groups) == 1:
            param_groups[0]["params"] = self.module.parameters()
        else:
            raise NotImplementedError
        return param_groups

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super(Model, self).state_dict()
        # TODO check copy of torch module, deepcopy?
        state_dict.update(
            device=str(self.device),
            module_state_dict=self.module.state_dict().copy(),
            state=self.state,
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        device = state_dict.pop("device")
        module_state_dict = state_dict.pop("module_state_dict")
        state = state_dict.pop("state")

        super(Model, self).load_state_dict(state_dict)

        self.device = device
        self.module.load_state_dict(module_state_dict)
        self.module.to(self.device)
        self.state = state


@loads(None)
class ModelConfig(SeededModuleConfig):
    module_load_file: Optional[Path] = None
    device: Union[str, List[str]] = "cpu"
    output_device: Optional[str] = None
    state_name_dataset: str = "dataset_state"
    state_name_model: str = "model_state"
    force_gradient: Optional[bool] = None
    compile_model: bool = False
    data_parallel: Optional[Literal["DP", "DDP"]] = None


class IdentityModel(Model):
    def __init__(self, **kwargs):
        torch_model = nn.Identity()
        super().__init__(torch_model, **kwargs)


@loads(IdentityModel)
class IdentityModelConfig(ModelConfig):
    pass


class DDPSetup(Module):
    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: Optional[str] = None,
        master_address: Optional[str] = None,
        master_port: Optional[str] = None,
        set_device: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.master_address = master_address
        self.master_port = master_port

        if set_device:
            torch.cuda.set_device(rank)

        if master_address is not None:
            os.environ["MASTER_ADDR"] = master_address
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)

        distributed.init_process_group(backend, rank=rank, world_size=world_size)

    def teardown(self, state: State) -> None:
        distributed.destroy_process_group()


@loads(DDPSetup)
class DDPSetupConfig(ModuleConfig):
    name: str = "DDPSetup"
    backend: Optional[str] = None
    rank: int
    world_size: int
    master_address: Optional[str] = "localhost"
    master_port: Optional[str | int] = None
    set_device: bool = True
