from typing import Callable, List

from core.df_detection.mri_gan.data_utils.utils import \
    get_dfdc_training_video_filepaths
from core.df_detection.mri_gan.utils import ConfigParser
from enums import DATA_TYPE


class MRIGANWorker:

    def __init__(self, data_type: DATA_TYPE) -> None:
        """Simple class with convenient functions for getting data paths for
        DFDC dataset based on the `data_type` argument.

        Parameters
        ----------
        data_type : DATA_TYPE
            on what kind of data type is worker working on
        """
        self._data_type = data_type

    @staticmethod
    def _get_fun_by_name_from_config(name: str) -> Callable:
        return getattr(ConfigParser.getInstance(), name)

    def _get_dfdc_data_path(self) -> str:
        return MRIGANWorker._get_fun_by_name_from_config(
            f'get_dfdc_{self._data_type.value}_data_path'
        )()

    def _get_dfdc_landmarks_data_path(self) -> str:
        return MRIGANWorker._get_fun_by_name_from_config(
            f'get_dfdc_landmarks_{self._data_type.value}_path'
        )()

    def _get_data_paths(self) -> List[str]:
        data_path_root = self._get_dfdc_data_path()
        file_paths = get_dfdc_training_video_filepaths(data_path_root)
        return file_paths

    def _get_dfdc_landmarks_data_paths(self) -> str:
        return MRIGANWorker._get_fun_by_name_from_config(
            f'get_dfdc_landmarks_{self._data_type.value}_path'
        )()

    def _get_dfdc_crops_data_path(self) -> str:
        return MRIGANWorker._get_fun_by_name_from_config(
            f'get_dfdc_crops_{self._data_type.value}_path'
        )()

    def _get_dfdc_mri_medatata_csv_path(self) -> str:
        return ConfigParser.getInstance().get_dfdc_mri_metadata_csv_path()

    def _get_dfdc_mri_path(self) -> str:
        return ConfigParser.getInstance().get_dfdc_mri_path()

    def _get_blank_image_path(self) -> str:
        return ConfigParser.getInstance().get_blank_image_path()

    def _get_mri_train_real_dataset_csv_path(self) -> str:
        return ConfigParser.getInstance().get_mri_train_real_dataset_csv_path()

    def _get_mri_test_real_dataset_csv_path(self) -> str:
        return ConfigParser.getInstance().get_mri_test_real_dataset_csv_path()

    def _get_mri_train_fake_dataset_csv_path(self) -> str:
        return ConfigParser.getInstance().get_mri_train_fake_dataset_csv_path()

    def _get_mri_test_fake_dataset_csv_path(self) -> str:
        return ConfigParser.getInstance().get_mri_test_fake_dataset_csv_path()
