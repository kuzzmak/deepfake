from dataclasses import dataclass
from enums import IO_OP_TYPE


@dataclass
class IO_OP:
    type: IO_OP_TYPE
    value: str
