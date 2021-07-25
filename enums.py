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
    SAVE = 'save'


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
    FRAMES_EXTRACTION_WORKER = 'frames_extraction_workers'
    IO_WORKER = 'io_worker'
    JOB_PROGRESS = 'job_progress'
    CONFIGURE_WIDGET = 'configure_widget'
    MESSAGE_WORKER = 'message_worker'


class MESSAGE_TYPE(Enum):
    REQUEST = 'request'
    ANSWER = 'answer'


class MESSAGE_STATUS(Enum):
    OK = 'ok'
    FAILED = 'failed'


class JOB_TYPE(Enum):
    IO_OPERATION = 'io_operation'
    CONSOLE_PRINT = 'console_print'
    FRAME_EXTRACTION = 'frame_extraction'
    WIDGET_CONFIGURATION = 'widget_configuration'
    NO_JOB = 'no_job'


class APP_STATUS(Enum):
    BUSY = 'BUSY'
    NO_JOB = 'NO JOB'


class WIDGET(Enum):
    JOB_PROGRESS = 'job_progress'
