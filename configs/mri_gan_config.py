import os
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Union

import yaml

from variables import DATA_ROOT, MRI_GAN_CONFIG_PATH


class MRIGANConfig:
    __instance = None

    @staticmethod
    def getInstance():
        if MRIGANConfig.__instance is None:
            MRIGANConfig()
        return MRIGANConfig.__instance

    def __init__(self):
        if MRIGANConfig.__instance is not None:
            raise Exception('ConfigParser class is a singleton!')
        else:
            MRIGANConfig.__instance = self

        if not MRI_GAN_CONFIG_PATH.exists():
            generate_default_mri_gan_config()

        with open(MRI_GAN_CONFIG_PATH, 'r') as f:
            self._config = yaml.safe_load(f)

        self.create_placeholders()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, config: Dict[str, Any]) -> None:
        self._config = config

    def save(self) -> None:
        save_yaml_config(self.config, MRI_GAN_CONFIG_PATH)

    def print_config(self):
        pprint(self._config)

    def get_log_dir_name(self, create_logdir=True):
        log_dir = os.path.join(
            self._config['logging']['root_log_dir'],
            self.init_time_str,
        )
        if create_logdir:
            os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def get_assets_path(self):
        return self._config['assets']

    def get_dfdc_train_data_path(self):
        return self._config['data_path']['dfdc']['train']

    def get_dfdc_valid_data_path(self):
        return self._config['data_path']['dfdc']['valid']

    def get_dfdc_test_data_path(self):
        return self._config['data_path']['dfdc']['test']

    def get_dfdc_train_label_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['data_path']['dfdc']
                            ['train_labels_csv_filename'])

    def get_dfdc_valid_label_csv_path(self):
        return os.path.join(self.get_dfdc_valid_data_path(),
                            self._config['data_path']['dfdc']
                            ['valid_labels_csv_filename'])

    def get_dfdc_test_label_csv_path(self):
        return os.path.join(self.get_dfdc_test_data_path(),
                            self._config['data_path']['dfdc']
                            ['test_labels_csv_filename'])

    def get_dfdc_train_frame_label_csv_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['data_path']['dfdc']
            ['train_frame_labels_csv_filename'])

    def get_dfdc_valid_frame_label_csv_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['data_path']['dfdc']
            ['valid_frame_labels_csv_filename'])

    def get_dfdc_test_frame_label_csv_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['data_path']['dfdc']
            ['test_frame_labels_csv_filename'])

    def get_data_aug_plan_pkl_filename(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['data_path']['dfdc']['data_augmentation']
            ['plan_pkl_filename'])

    def get_aug_metadata_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['data_path']['dfdc']['data_augmentation']
            ['metadata'])

    def get_dfdc_landmarks_train_path(self):
        return self._config['features']['dfdc']['landmarks_path']['train']

    def get_dfdc_landmarks_valid_path(self):
        return self._config['features']['dfdc']['landmarks_path']['valid']

    def get_dfdc_landmarks_test_path(self):
        return self._config['features']['dfdc']['landmarks_path']['test']

    def get_dfdc_crops_train_path(self):
        return self._config['features']['dfdc']['crop_faces']['train']

    def get_dfdc_crops_valid_path(self):
        return self._config['features']['dfdc']['crop_faces']['valid']

    def get_dfdc_crops_test_path(self):
        return self._config['features']['dfdc']['crop_faces']['test']

    def get_train_mrip2p_png_data_path(self):
        return self._config['features']['dfdc']['train_mrip2p_faces']

    def get_valid_mrip2p_png_data_path(self):
        return self._config['features']['dfdc']['valid_mrip2p_faces']

    def get_test_mrip2p_png_data_path(self):
        return self._config['features']['dfdc']['test_mrip2p_faces']

    def get_train_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']['dfdc']
                            ['train_mriframe_label'])

    def get_valid_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']['dfdc']
                            ['valid_mriframe_label'])

    def get_test_mriframe_label_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']['dfdc']
                            ['test_mriframe_label'])

    def get_dfdc_mri_metadata_csv_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['features']['dfdc']['mri_metadata_csv'])

    def get_dfdc_mri_path(self):
        return self._config['features']['dfdc']['mri_path']

    def get_mri_train_real_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']
                            ['mri_dataset_real_train_csv'])

    def get_mri_train_fake_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']
                            ['mri_dataset_fake_train_csv'])

    def get_mri_test_real_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']
                            ['mri_dataset_real_test_csv'])

    def get_mri_test_fake_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']
                            ['mri_dataset_fake_test_csv'])

    def get_mri_dataset_csv_path(self):
        return os.path.join(self.get_assets_path(),
                            self._config['features']['mri_dataset_csv'])

    def get_blank_image_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['features']['blank_png'])

    def get_mri_gan_weight_path(self):
        return os.path.join(
            self.get_assets_path(),
            self._config['MRI_GAN']['weights'])

    def get_mri_gan_model_params(self):
        return self._config['MRI_GAN']['model_params']

    def get_default_cnn_encoder_name(self):
        return self._config['cnn_encoder']['default']

    def get_training_sample_size(self):
        return float(self._config['deep_fake']['training']['train_size'])

    def get_valid_sample_size(self):
        return float(self._config['deep_fake']['training']['valid_size'])

    def get_test_sample_size(self):
        return float(self._config['deep_fake']['training']['test_size'])

    def get_deep_fake_training_params(self):
        return self._config['deep_fake']['training']['model_params']

    def get_log_params(self):
        return self._config['logging']

    def create_placeholders(self):
        os.makedirs(self.get_assets_path(), exist_ok=True)


def print_line():
    print('-' * MRIGANConfig.getInstance()._config['logging']['line_len'])


def print_green(text):
    """
    print text in green color
    @param text: text to print
    """
    print('\033[32m', text, '\033[0m', sep='')


def print_red(text):
    """
    print text in green color
    @param text: text to print
    """
    print('\033[31m', text, '\033[0m', sep='')


def save_yaml_config(config, config_path: Path) -> None:
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


def _get_default_config(
    base: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    if base is None:
        base = DATA_ROOT / 'df_dataset'
    elif isinstance(base, str):
        base = Path(base)

    dataset = Path('dfdc')
    lmrks = dataset / 'landmarks'
    crop_faces = dataset / 'crop_faces'

    return {
        'assets': 'assets',
        'data_path': {
            'dfdc': {
                'train': str(base / 'train'),
                'valid': str(base / 'valid'),
                'test': str(base / 'test'),
                'train_labels_csv_filename': 'train_labels.csv',
                'valid_labels_csv_filename': 'labels.csv',
                'test_labels_csv_filename': 'labels.csv',
                'train_frame_labels_csv_filename': 'train_frame_labels.csv',
                'valid_frame_labels_csv_filename': 'valid_frame_labels.csv',
                'test_frame_labels_csv_filename': 'test_frame_labels.csv',
                'data_augmentation': {
                    'plan_pkl_filename': 'data_augmentation_plan.pkl',
                    'metadata': 'aug_metadata',
                },
            }
        },
        'features': {
            'dfdc': {
                'landmarks_path': {
                    'train': str(base / lmrks / 'train'),
                    'valid': str(base / lmrks / 'valid'),
                    'test': str(base / lmrks / 'test'),
                },
                'crop_faces': {
                    'train': str(base / crop_faces / 'train'),
                    'valid': str(base / crop_faces / 'valid'),
                    'test': str(base / crop_faces / 'test'),
                },
                'mri_path': str(base / dataset / 'mri'),
                'mri_metadata_csv': 'dfdc_mri_metadata.csv',
                'train_mriframe_label': 'train_mriframe_labels.csv',
                'valid_mriframe_label': 'valid_mriframe_labels.csv',
                'test_mriframe_label': 'test_mriframe_labels.csv',
            },
            'mri_dataset_real_train_csv': 'mri_real_train_dataset.csv',
            'mri_dataset_fake_train_csv': 'mri_fake_train_dataset.csv',
            'mri_dataset_real_test_csv': 'mri_real_test_dataset.csv',
            'mri_dataset_fake_test_csv': 'mri_fake_test_dataset.csv',
            'mri_dataset_csv': 'mri_dataset.csv',
            'blank_png': 'blank.png',
        },
        'MRI_GAN': {
            'weights': 'weights/MRI_GAN_weights.chkpt',
            'model_params': {
                'n_epochs': 100,
                'batch_size': 128,
                'lambda_pixel': 100,
                'b1': 0.5,
                'b2': 0.999,
                'lr': 0.0002,
                'tau': 0.3,
                'imsize': 256,
                'model_name': 'MRI_GAN',
                'test_sample_size': 16,
                'chkpt_freq': 500,
                'sample_gen_freq': 200,
                'frac': 1,
                'losses_file': 'losses.pkl',
                'metadata_file': 'mri_metadata.pkl',
                'ssim_report_file': 'ssim_report_file.pkl',
            },
        },
        'cnn_encoder': {
            'default': 'tf_efficientnet_b0_ns'
        },
        'deep_fake': {
            'training': {
                'train_size': 1,
                'valid_size': 1,
                'test_size': 1,
                'model_params': {
                    'model_name': 'DeepFakeDetectModel',
                    'label_smoothing': 0.1,
                    'train_transform': 'simple',  # simple or complex
                    'batch_format': 'simple',
                    'epochs': 20,
                    'learning_rate': 0.001,
                    'batch_size': 192,
                    'fp16': True,
                    'opt_level': 'O0',
                    'dataset': 'mri',
                },
            },
        },
        'logging': {
            'root_log_dir': 'logs',
            'line_len': 80,
            'model_info_log': 'model_info_and_results.log',
            'model_loss_info_log': 'model_losses.log',
            'model_acc_info_log': 'model_acc.log',
            'model_conf_matrix_csv': 'confusion_matrix.csv',
            'model_conf_matrix_png': 'confusion_matrix.png',
            'model_conf_matrix_normalized_csv':
                'confusion_matrix_normalized.csv',
            'model_conf_matrix_normalized_png':
                'confusion_matrix_normalized.png',
            'model_accuracy_png': 'model_accuracy.png',
            'model_loss_png': 'model_loss.png',
            'all_samples_pred_csv': 'all_samples_pred.csv',
            'model_roc_png': 'model_roc.png',
        },
    }


def generate_default_mri_gan_config(
    base: Optional[Union[str, Path]] = None
) -> None:
    save_yaml_config(_get_default_config(base), MRI_GAN_CONFIG_PATH)
