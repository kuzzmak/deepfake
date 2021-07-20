from dataclasses import dataclass
from enum import Enum


class MESSAGE_TYPE(Enum):
    IO_OPERATION = 'io_operation'
    CONSOLE_PRINT = 'console_print'


@dataclass
class Message:
    type: MESSAGE_TYPE
    contents: dict()

    def get_data(self):
        if self.type == MESSAGE_TYPE.IO_OPERATION:
            op = self.contents['operation_type']
            file = self.contents['file']
            return op, file
