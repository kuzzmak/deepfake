from pathlib import Path

from enums import VIDEO_FORMAT

DEEPFAKE_ROOT = Path(__file__).parent.absolute()

DATA_ROOT = DEEPFAKE_ROOT / 'data'

RESOURCES_ROOT = DEEPFAKE_ROOT / 'resources'

RESOURCE_IMAGES_ROOT = RESOURCES_ROOT / 'images'

CONFIGS_ROOT_PATH = DEEPFAKE_ROOT / 'configs'

MRI_GAN_CONFIG_PATH = CONFIGS_ROOT_PATH / 'mri_gan_config.yaml'

APP_CONFIG_PATH = CONFIGS_ROOT_PATH / 'app_config.json'

LOGGING_CONFIG_PATH = CONFIGS_ROOT_PATH / 'logging_config.yaml'

LOGS_PATH = DEEPFAKE_ROOT / 'logs'

LONG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LONG_DATE_FORMAT_FILE_NAME = '%Y_%m_%d_%H_%M_%S'

ETA_FORMAT = 'ETA: {}'

SUPPORTED_VIDEO_EXTS = set(['.' + vf.value for vf in VIDEO_FORMAT])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

APP_NAME = 'Deepfake tool'

START_PAGE_NAME = 'start_page'
START_PAGE_TITLE = 'Start page'

MAKE_DEEPFAKE_PAGE_NAME = 'make_deepfake_page'
MAKE_DEEPFAKE_PAGE_TITLE = 'Make deepfake page'

DETECT_DEEPFAKE_PAGE_NAME = 'detect_deepfake_page'
DETECT_DEEPFAKE_PAGE_TITLE = 'Detect deepfake page'
