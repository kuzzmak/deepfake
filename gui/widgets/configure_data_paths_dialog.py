import logging
from pathlib import Path
from typing import Callable, List

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.mri_gan_config import MRIGANConfig
from gui.widgets.common import Button, HWidget, VWidget, VerticalSpacer
from utils import get_val_from_dict, set_val_on_dict


def _get_conf_value_by_key(key: List[str]) -> str:
    return get_val_from_dict(MRIGANConfig.get_instance().config, key)


logger = logging.getLogger(__name__)


class ConfigureDataPathsDialog(qwt.QDialog):

    dfdc_data_path = ['data_path', 'dfdc']
    dfdc_data_path_train = [*dfdc_data_path, 'train']
    dfdc_data_path_valid = [*dfdc_data_path, 'valid']
    dfdc_data_path_test = [*dfdc_data_path, 'test']
    dfdc_data_path_all = [
        dfdc_data_path_train,
        dfdc_data_path_valid,
        dfdc_data_path_test,
    ]

    dfdc_features = ['features', 'dfdc']

    dfdc_features_landmarks_path = [*dfdc_features, 'landmarks_path']
    dfdc_landmarks_path_train = [*dfdc_features_landmarks_path, 'train']
    dfdc_landmarks_path_valid = [*dfdc_features_landmarks_path, 'valid']
    dfdc_landmarks_path_test = [*dfdc_features_landmarks_path, 'test']
    dfdc_landmarks_path_all = [
        dfdc_landmarks_path_train,
        dfdc_landmarks_path_valid,
        dfdc_landmarks_path_test,
    ]

    dfdc_crop_faces_path = [*dfdc_features, 'crop_faces']
    dfdc_crop_faces_path_train = [*dfdc_crop_faces_path, 'train']
    dfdc_crop_faces_path_valid = [*dfdc_crop_faces_path, 'valid']
    dfdc_crop_faces_path_test = [*dfdc_crop_faces_path, 'test']
    dfdc_crop_faces_path_all = [
        dfdc_crop_faces_path_train,
        dfdc_crop_faces_path_valid,
        dfdc_crop_faces_path_test,
    ]

    mri_metadata_csv_path = [*dfdc_features, 'mri_metadata_csv']

    dfdc_mri_path = [*dfdc_features, 'mri_path']

    def __init__(self, keys: List[List[str]], labels: List[str]):
        super().__init__()

        self._keys = keys
        self._labels = labels

        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle('Configure data paths')

        self.button_box = qwt.QDialogButtonBox()
        self.save_btn = self.button_box.addButton(
            'save',
            qwt.QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.save_btn.clicked.connect(self._save_mri_gan_config)
        self.cancel_btn = self.button_box.addButton(
            'cancel',
            qwt.QDialogButtonBox.ButtonRole.RejectRole,
        )
        self.cancel_btn.clicked.connect(self.reject)

        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        for key, label in zip(self._keys, self._labels):
            path_row = HWidget()
            layout.addWidget(path_row)
            label_row = VWidget()
            path_row.layout().addWidget(label_row)
            label_row.layout().setContentsMargins(0, 0, 0, 0)
            label_row.layout().addWidget(qwt.QLabel(text=label))
            lbl_name = ConfigureDataPathsDialog. \
                _construct_label_name_from_key(key)
            setattr(
                self,
                lbl_name,
                qwt.QLabel(text=_get_conf_value_by_key(key)),
            )
            label_row.layout().addWidget(
                getattr(self, lbl_name)
            )
            select_btn = Button('select', 100)
            path_row.layout().addWidget(select_btn)
            select_btn.clicked.connect(self._select_path(key))

        layout.addItem(VerticalSpacer())
        layout.addWidget(self.button_box)

    @qtc.pyqtSlot()
    def _select_path(self, key: List[str]) -> Callable:
        """Prompts user with the dialog to choose directory for a particular
        path.

        Parameters
        ----------
        key : List[str]
            key for which is path being selected
        """
        def __select_path():
            dir = str(qwt.QFileDialog.getExistingDirectory(
                self,
                'Select directory path',
            ))
            if not dir:
                logger.warning('No directory selected.')
                return
            self._change_current_path(key, dir)
        return __select_path

    @staticmethod
    def _construct_label_name_from_key(key: List[str]) -> str:
        """Constructs name of the label based on the `key`.

        Parameters
        ----------
        key : List[str]
            key, list of keys in config dict

        Returns
        -------
        str
            constructed name
        """
        return '_'.join([*key, 'lbl'])

    def _change_current_path(self, key: List[str], path: str) -> None:
        """Updates some path in dialog based on the new path user chose.

        Parameters
        ----------
        key : List[str]
            key for determining which path to update
        path : str
            new value of the path
        """
        lbl_name = ConfigureDataPathsDialog._construct_label_name_from_key(key)
        lbl: qwt.QLabel = getattr(self, lbl_name)
        lbl.setText(path)

    def _save_mri_gan_config(self) -> None:
        """Goes through all paths in current dialog and saves them.
        """
        conf = MRIGANConfig.get_instance().config
        for key in self._keys:
            lbl_key = ConfigureDataPathsDialog. \
                _construct_label_name_from_key(key)
            lbl: qwt.QLabel = getattr(self, lbl_key)
            lbl_text = str(Path(lbl.text()))
            set_val_on_dict(conf, key, lbl_text)
        MRIGANConfig.get_instance().config = conf
        MRIGANConfig.get_instance().save()
        self.close()
