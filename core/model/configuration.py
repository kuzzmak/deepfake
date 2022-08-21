from dataclasses import dataclass, field
from typing import Any, Dict

from torch.nn import Module

from enums import MODEL


@dataclass
class ModelConfiguration:
    model: MODEL
    model_args: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    model: Module
    options: Dict[str, Any] = field(default_factory=dict)
