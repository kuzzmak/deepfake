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


class IO_OPERATION_TYPE(Enum):
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
    NO_OWNER = 'no_owner'

    # action signals
    CONFIGURE_WIDGET = 'configure_widget'
    FRAMES_EXTRACTION = 'frames_extraction'
    INPUT_DATA_DIRECTORY = 'input_data_directory'
    OUTPUT_DATA_DIRECTORY = 'output_data_directory'

    MAKE_DEEPFAKE_PAGE_DETECT_FACES = 'make_deepfake_page_detect_faces'

    # pages
    MAKE_DEEPFAKE_PAGE = 'make_deepfake_page'

    # specific widget signals
    CONSOLE = 'console'
    JOB_PROGRESS = 'job_progress'
    DATA_SELECTOR = 'data_selector'
    DETECTION_ALGORITHM_TAB = 'detection_algorithm_tab'
    DETECTION_ALGORITHM_TAB_INPUT_PICTURE_VIEWER = \
        'detection_algorithm_tab_input_picture_viewer'
    DETECTION_ALGORITHM_TAB_OUTPUT_PICTURE_VIEWER = \
        'detection_algorithm_tab_output_picture_viewer'

    # worker signals
    IO_WORKER = 'io_worker'
    MESSAGE_WORKER = 'message_worker'
    FACE_DETECTION_WORKER = 'face_detection_worker'
    FRAMES_EXTRACTION_WORKER = 'frames_extraction_workers'


class MESSAGE_TYPE(Enum):
    REQUEST = 'request'
    ANSWER = 'answer'


class MESSAGE_STATUS(Enum):
    OK = 'ok'
    FAILED = 'failed'


class DATA_TYPE(Enum):
    INPUT = 'Input'
    OUTPUT = 'Output'


class IMAGE_FORMATS(Enum):
    PNG = 'png'
    JPG = 'jpg'


class JOB_TYPE(Enum):
    IO_OPERATION = 'io_operation'
    CONSOLE_PRINT = 'console_print'
    FRAME_EXTRACTION = 'frame_extraction'
    WIDGET_CONFIGURATION = 'widget_configuration'
    FACE_DETECTION = 'face_detection'
    IMAGE_DISPLAY = 'image_display'
    ADD_SIGNAL = 'add_dignal'
    NO_JOB = 'no_job'


class APP_STATUS(Enum):
    BUSY = 'BUSY'
    NO_JOB = 'NO JOB'


class WIDGET(Enum):
    JOB_PROGRESS = 'job_progress'


class FACE_DETECTION_ALGORITHM(Enum):
    S3FD = 's3fd'
    MTCNN = 'mtcnn'
    FACEBOXES = 'faceboxes'


class FILE_TYPE(Enum):
    IMAGE = 'image'


class BODY_KEY(Enum):
    IO_OPERATION_TYPE = 'io_operation_type'
    INPUT_DATA_DIRECTORY = 'input_data_directory'
    OUTPUT_DATA_DIRECTORY = 'output_data_directory'
    INPUT_FACES_DIRECTORY = 'input_faces_directory'
    OUTPUT_FACES_DIRECTORY = 'output_faces_directory'
    FILE_PATH = 'file_path'
    NEW_FILE_PATH = 'new_file_path'
    FILE = 'file'
    FILE_TYPE = 'file_type'
    RESIZE = 'resize'
    NEW_SIZE = 'new_size'
    MULTIPART = 'multipart'
    PART = 'part'
    TOTAL = 'total'
    VIDEO_PATH = 'video_path'
    DATA_DIRECTORY = 'data_directory'
    DATA_TYPE = 'data_type'
    WIDGET = 'widget'
    METHOD = 'method'
    ARGS = 'args'
    CONSOLE_MESSAGE_TYPE = 'console_message_type'
    MESSAGE = 'message'
    SIGNAL = 'signal'
    SIGNAL_OWNER = 'signal_owner'
    MODEL_PATH = 'model_path'
    ALGORITHM = 'algorithm'
