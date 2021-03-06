from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

import torch
from baselooper import SeededModule, State, SeededModuleConfig
from torch import nn

from mllooper.data import DatasetState


@dataclass
class ModelState(State):
    output: Optional[Any] = None


class Model(SeededModule, ABC):
    def __init__(self, torch_model: nn.Module, module_load_file: Optional[Path] = None, device: str = 'cpu', **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.module = torch_model.to(self.device)

        if module_load_file:
            module_state_dict = torch.load(module_load_file, map_location=self.device)
            self.module.load_state_dict(module_state_dict)

        self.state = ModelState()

    def step(self, state: State) -> None:
        dataset_state: DatasetState = state.dataset_state
        self.state.output = None

        module_input = self.format_module_input(dataset_state.data)
        with torch.set_grad_enabled(dataset_state.train):
            module_output = self.module(module_input)

        self.state.output = self.format_module_output(module_output)
        state.model_state = self.state

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
        state_dict.update(
            device=str(self.device),
            module_state_dict=self.module.state_dict(),
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


class ModelConfig(SeededModuleConfig):
    module_load_file: Optional[Path]
