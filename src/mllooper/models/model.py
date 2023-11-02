from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

import torch
from torch import nn
from yaloader import loads

from mllooper import SeededModule, State, SeededModuleConfig
from mllooper.data import DatasetState


@dataclass
class ModelState(State):
    output: Optional[Any] = None


class Model(SeededModule, ABC):
    def __init__(self, torch_model: nn.Module, module_load_file: Optional[Path] = None,
                 device: Union[str, List[str]] = 'cpu',
                 output_device: Optional[str] = None,
                 dataset_state_name: str = 'dataset_state',
                 model_state_name: str = 'model_state',
                 force_gradient: Optional[bool] = None,
                 **kwargs):
        super().__init__(**kwargs)
        devices = device if isinstance(device, list) else [device]
        self.devices = [torch.device(device) for device in devices]
        self.device = self.devices[0]
        self.output_device = output_device
        self.module = torch_model.to(self.device)
        self._parallel_module = self.module if len(self.devices) == 1 else nn.DataParallel(self.module, device_ids=self.devices, output_device=output_device)
        self.dataset_state_name: str = dataset_state_name
        self.model_state_name: str = model_state_name

        if module_load_file:
            module_state_dict = torch.load(module_load_file, map_location=self.device)
            self.module.load_state_dict(module_state_dict)

        self.force_gradient: Optional[bool] = force_gradient
        self.state = ModelState()

    def step(self, state: State) -> None:
        dataset_state: DatasetState = getattr(state, self.dataset_state_name)
        self.state.output = None

        module_input = self.format_module_input(dataset_state.data)

        self.module.train() if dataset_state.train else self.module.eval()
        self._parallel_module.train() if dataset_state.train else self._parallel_module.eval()
        with torch.set_grad_enabled(self.force_gradient if self.force_gradient is not None else dataset_state.train):
            # module_output = self.module(module_input)
            module_output = self._parallel_module(module_input)

        self.state.output = self.format_module_output(module_output)
        setattr(state, self.model_state_name, self.state)

    @staticmethod
    def format_module_input(data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, Dict) and 'input' in data.keys() and isinstance(data['input'], torch.Tensor):
            return data['input']
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
            return [{'params': self.module.parameters()}]
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
            state=self.state
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        device = state_dict.pop('device')
        module_state_dict = state_dict.pop('module_state_dict')
        state = state_dict.pop('state')

        super(Model, self).load_state_dict(state_dict)

        self.device = device
        self.module.load_state_dict(module_state_dict)
        self.module.to(self.device)
        self.state = state


@loads(None)
class ModelConfig(SeededModuleConfig):
    module_load_file: Optional[Path] = None
    device: Union[str, List[str]] = 'cpu'
    output_device: Optional[str] = None
    dataset_state_name: str = 'dataset_state'
    model_state_name: str = 'model_state'
    force_gradient: Optional[bool] = None
