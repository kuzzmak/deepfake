from typing import Callable, List

from core.df_detection.mri_gan.data_utils.utils import \
    get_dfdc_training_video_filepaths
from core.df_detection.mri_gan.utils import ConfigParser
from enums import DATA_TYPE


class MRIGANWorker:

    def __init__(self, data_type: DATA_TYPE) -> None:
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
