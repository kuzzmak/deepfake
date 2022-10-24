from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, Union

from torch.optim import Adam
from enums import OPTIMIZER


@dataclass
class OptimizerConfiguration:
    optimizer_class: OPTIMIZER
    optimizer_module: str
    args: dict = field(default_factory=dict)

    def values(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        return {
            'optimizer_module': self.optimizer_module,
            'optimizer_class': self.optimizer_class.value,
            'options': self.args,
        }


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
