from pathlib import Path

from enums import VIDEO_FORMAT

DEEPFAKE_ROOT = Path(__file__).parent.absolute()

DATA_ROOT = DEEPFAKE_ROOT / 'data'

RESOURCES_ROOT = DEEPFAKE_ROOT / 'resources'

RESOURCE_IMAGES_ROOT = RESOURCES_ROOT / 'images'

CONFIGS_ROOT_PATH = DEEPFAKE_ROOT / 'configs'

MRI_GAN_CONFIG_PATH = CONFIGS_ROOT_PATH / 'mri_gan_config.yaml'

LONG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LONG_DATE_FORMAT_FILE_NAME = '%Y_%m_%d_%H_%M_%S'

ETA_FORMAT = 'ETA: {}'

SUPPORTED_VIDEO_EXTS = set(['.' + vf.value for vf in VIDEO_FORMAT])
