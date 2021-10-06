from dataclasses import dataclass, field

from enums import OPTIMIZER


@dataclass
class OptimizerConfiguration:
    optimizer: OPTIMIZER
    optimizer_args: dict = field(default_factory=dict)


DEFAULT_ADAM_CONF = OptimizerConfiguration(
    OPTIMIZER.ADAM,
    dict(
        {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
        }
    ),
)
