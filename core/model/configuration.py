from dataclasses import dataclass, field

from enums import MODEL


@dataclass
class ModelConfiguration:
    model: MODEL
    model_args: dict = field(default_factory=dict)
