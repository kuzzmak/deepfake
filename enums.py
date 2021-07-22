from collections import namedtuple
from enum import Enum

ConsolePrefix = namedtuple('ConsolePrefix', 'prefix prefix_color')


class CONSOLE_COLORS(Enum):
    RED = '#ff0000'
    BLACK = '#000000'
    ORANGE = '#ffa500'


class CONSOLE_MESSAGE_TYPE(Enum):
    LOG = ConsolePrefix('[LOG]', CONSOLE_COLORS.BLACK)
    INFO = ConsolePrefix('[INFO]', CONSOLE_COLORS.BLACK)
    ERROR = ConsolePrefix('[ERROR]', CONSOLE_COLORS.RED)
    WARNING = ConsolePrefix('[WARNING]', CONSOLE_COLORS.ORANGE)


class IO_OP_TYPE(Enum):
    DELETE = 'delete'
    RENAME = 'rename'


class DIALOG_MESSAGE_ICON(Enum):
    DELETE = ':/delete.svg'
    RENAME = ':/rename.svg'
    WARNING = ':/warning.svg'


class DIALOG_MESSAGE_TYPE(Enum):
    DELETE = 'Delete'
    RENAME = 'Rename'
    WARNING = 'Warning'


class SIGNAL_OWNER(Enum):
    CONOSLE = 'console'
