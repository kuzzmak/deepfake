from pathlib import Path

DEEPFAKE_ROOT = Path(__file__).parent.absolute()

DATA_ROOT = DEEPFAKE_ROOT / 'data'

RESOURCES_ROOT = DEEPFAKE_ROOT / 'resources'

RESOURCE_IMAGES_ROOT = RESOURCES_ROOT / 'images'

CONFIGS_ROOT_PATH = DEEPFAKE_ROOT / 'configs'

MRI_GAN_CONFIG_PATH = CONFIGS_ROOT_PATH / 'mri_gan_config.yaml'

LONG_DATE_FORMAt = '%Y-%m-%d %H:%M:%S'
