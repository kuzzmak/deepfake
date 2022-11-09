from __future__ import annotations
from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, Union

from torch.optim import Adam
from enums import OPTIMIZER


@dataclass
class OptimizerConfiguration:
    classname: OPTIMIZER
    module: str
    args: dict = field(default_factory=dict)

    def values(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        return {
            'class': self.classname.value,
            'module': self.module,
            'args': self.args,
        }

    @classmethod
    def from_dict(
        cls,
        dict: Dict[str, Union[str, Dict[str, Any]]],
    ) -> OptimizerConfiguration:
        return cls(
            classname=OPTIMIZER[dict['class'].upper()],
            module=dict['module'],
            args=dict['args'],
        )


def _default_arg_names_and_vals(cls):
    return {
        p.name: p.default for p in inspect.signature(cls).parameters.values()
        if p.default != inspect.Parameter.empty
        and p.kind == p.POSITIONAL_OR_KEYWORD
    }


DEFAULT_ADAM_CONF = OptimizerConfiguration(
    OPTIMIZER.ADAM,
    'torch.optim',
    _default_arg_names_and_vals(Adam),
)
