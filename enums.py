from collections import namedtuple
from enum import Enum

ConsolePrefix = namedtuple('ConsolePrefix', 'prefix prefix_color')


class CONSOLE_COLORS(Enum):
    RED = '#ff0000'
    BLACK = '#000000'
    ORANGE = '#ffa500'
    WHITE = '#ffffff'


class CONSOLE_MESSAGE_TYPE(Enum):
    LOG = ConsolePrefix('[LOG]', CONSOLE_COLORS.WHITE)
    ERROR = ConsolePrefix('[ERROR]', CONSOLE_COLORS.RED)
    WARNING = ConsolePrefix('[WARNING]', CONSOLE_COLORS.ORANGE)


class LEVEL(Enum):
    DEBUG = 'DEBUG'
    CRITICAL = 'CRITICAL'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'


class COLOR(Enum):
    RED = '#ff0000'
    BLACK = '#000000'
    ORANGE = '#ffa500'
    WHITE = '#ffffff'


class LEVEL_COLOR(Enum):
    DEBUG = COLOR.WHITE
    CRITICAL = COLOR.RED
    INFO = COLOR.WHITE
    WARNING = COLOR.ORANGE
    ERROR = COLOR.RED


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


class WORKER(Enum):
    IO_WORKER = 'io_worker'
    MESSAGE_WORKER = 'message_worker'
    NEXT_ELEMENT_WORKER = 'next_element_worker'
    FRAMES_EXTRACTION_WORKER = 'frames_extraction_workers'
    LANDMARK_EXTRACTION_WORKER = 'landmark_extraction_worker'
    CROPPING_FACES_WORKER = 'cropping_faces_worker'
    GENERATE_MRI_DATASET_WORKER = 'generate_mri_dataset_worker'
    TRAIN_MRI_GAN_WORKER = 'train_mri_gan_worker'
    FACE_EXTRACTION_WORKER = 'face_extaction_worker'


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
    IMAGE_VIEWER = 'image_viewer'
    DETECTION_ALGORITHM_TAB = 'detection_algorithm_tab'
    DETECTION_ALGORITHM_TAB_INPUT_PICTURE_VIEWER = \
        'detection_algorithm_tab_input_picture_viewer'
    MRI_GAN_WIDGET = 'mri_gan_widget'

    # worker signals
    IO_WORKER = WORKER.IO_WORKER.value
    MESSAGE_WORKER = WORKER.MESSAGE_WORKER.value
    NEXT_ELEMENT_WORKER = WORKER.NEXT_ELEMENT_WORKER.value
    FRAMES_EXTRACTION_WORKER = WORKER.FRAMES_EXTRACTION_WORKER.value
    LANDMARK_EXTRACTION_WORKER = WORKER.LANDMARK_EXTRACTION_WORKER.value
    CROPPING_FACES_WORKER = WORKER.CROPPING_FACES_WORKER.value
    GENERATE_MRI_DATASET_WORKER = WORKER.GENERATE_MRI_DATASET_WORKER.value
    TRAIN_MRI_GAN_WORKER = WORKER.TRAIN_MRI_GAN_WORKER.value
    FACE_EXTRACTION_WORKER = WORKER.FACE_EXTRACTION_WORKER

    # worker signals next element
    FRAMES_EXTRACTION_WORKER_NEXT_ELEMENT = \
        'frames_extraction_worker_next_element'

    ALIGNER = 'aligner'

    # display signals
    SHOW_CONSOLE = 'show_console'
    SHOW_MENUBAR = 'show_menubar'
    SHOW_TOOLBAR = 'show_toolbar'


class MESSAGE_TYPE(Enum):
    REQUEST = 'request'
    ANSWER = 'answer'
    JOB_EXIT = 'job_exit'


class MESSAGE_STATUS(Enum):
    OK = 'ok'
    FAILED = 'failed'


class IMAGE_FORMAT(Enum):
    PNG = 'png'
    JPG = 'jpg'


class JOB_TYPE(Enum):
    IO_OPERATION = 'io_operation'
    CONSOLE_PRINT = 'console_print'
    FRAMES_EXTRACTION = 'frames_extraction'
    WIDGET_CONFIGURATION = 'widget_configuration'
    IMAGE_DISPLAY = 'image_display'
    ADD_SIGNAL = 'add_dignal'
    NEXT_ELEMENT = 'next_element'
    NO_JOB = 'no_job'
    LANDMARK_ALIGNMENT = 'landmark_alignment'
    LANDMARK_EXTRACTION = 'landmark_extraction'
    CROPPING_FACES = 'cropping_faces'
    GENERATE_MRI_DATASET = 'generate_mri_dataset'
    TRAIN_MRI_GAN = 'train_mri_gan'
    TRAIN_DF_DETECTOR = 'train_df_detector'
    IMAGE_SCRAPING = 'image_scraping'
    FACE_EXTRACTION = 'face_extraction'


class APP_STATUS(Enum):
    BUSY = 'BUSY'
    NO_JOB = 'NO JOB'


class WIDGET(Enum):
    JOB_PROGRESS = 'job_progress'


class FACE_DETECTION_ALGORITHM(Enum):
    S3FD = 's3fd'
    FACEBOXES = 'faceboxes'


class LANDMARK_DETECTION_ALGORITHM(Enum):
    FAN = 'fan'


class FILE_TYPE(Enum):
    IMAGE = 'image'


class DEVICE(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'


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
    ALGORITHM = 'algorithm'
    WORKER = 'worker'
    DEVICE = 'device'
    EVERY_N_TH_FRAME = 'every_n_th_frame'
    JOB_NAME = 'job_name'
    ETA = 'eta'


class MODEL(Enum):
    ORIGINAL = 'original'


class OPTIMIZER(Enum):
    ADAM = 'Adam'


class INTERPOLATION(Enum):
    NEAREST = 0
    LINEAR = 1
    CUBIC = 2
    AREA = 3


class INDEX_TYPE(Enum):
    QMODELINDEX = 'qmodelindex'
    INT = 'int'


class IMAGE_SORT(Enum):
    IMAGE_HASH = 'image_hash'


class NUMBER_TYPE(Enum):
    INT = 'int'
    FLOAT = 'float'


class MASK_DIM(Enum):
    ONE = 1
    THREE = 3


class LAYOUT(Enum):
    VERTICAL = 'vertical'
    HORIZONTAL = 'horizontal'


class DF_DETECTION_MODEL(Enum):
    MESO_NET = 'meso_net'
    MRI_GAN = 'mri_gan'


class CONNECTION(Enum):
    STOP = 'stop'


class DATA_TYPE(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'
    ALL = 'all'


class JOB_NAME(Enum):
    LANDMARK_EXTRACTION = 'landmark extraction'
    CROPPING_FACES = 'cropping faces'
    GENERATE_MRI_DATASET = 'generating mri dataset'
    TRAIN_MRI_GAN = 'training mri gan'
    FRAMES_EXTRACTION = 'extracting frames'
    LOADING = 'loading'
    ALIGNING_LANDMARKS = 'aligning landmarks'
    TRAIN_DF_DETECTOR = 'training df detector'
    IMAGE_SCRAPING = 'scraping images'
    FACE_EXTRACTION = 'extracting faces'


class WORKER_THREAD(Enum):
    MESSAGE_WORKER = 'message_worker_thread'
