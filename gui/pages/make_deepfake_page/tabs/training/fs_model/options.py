import logging
from pathlib import Path
from typing import Optional

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from core.model.fs import FS
from enums import FREQUENCY_UNIT, WIDGET_TYPE
from gui.pages.make_deepfake_page.tabs.training.widgets import (
    LoggingConfig,
    SelectDirRow,
)
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    GroupBox,
    HWidget,
    InfoIconButton,
    Parameter,
    RefreshIconButton,
    VerticalSpacer,
)
from utils import str_to_bool
from variables import APP_LOGGER

logger = logging.getLogger(APP_LOGGER)


class Options(BaseWidget):

    refresh_runs_sig = qtc.pyqtSignal()
    run_changed_sig = qtc.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self._model_name = FS.NAME

        self._init_ui()

        self.refresh_runs_sig.connect(self._refresh_runs)

    def _init_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        train_options_gb = GroupBox('Training options')
        layout.addWidget(train_options_gb)

        self._resume = Parameter(
            'resume',
            [True, False],
            False,
            WIDGET_TYPE.RADIO_BUTTON,
        )
        train_options_gb.layout().addWidget(self._resume)

        run_row = HWidget()
        train_options_gb.layout().addWidget(run_row)
        run_row.layout().setContentsMargins(0, 0, 0, 0)
        self._run = Parameter('run', [], None, WIDGET_TYPE.DROPDOWN)
        run_row.layout().addWidget(self._run)
        self._run._cb.currentIndexChanged.connect(self._run_selection_changed)
        self._run_info_btn = InfoIconButton()
        run_row.layout().addWidget(self._run_info_btn)
        self._run_info_btn.clicked.connect(self._run_info)
        self._refresh_runs_btn = RefreshIconButton()
        run_row.layout().addWidget(self._refresh_runs_btn)
        self._refresh_runs_btn.clicked.connect(self._refresh_runs)

        self._steps = Parameter('steps', [100000])
        train_options_gb.layout().addWidget(self._steps)

        self._use_cudnn_bench = Parameter(
            'use cudnn benchmark',
            [True, False],
            True,
            WIDGET_TYPE.RADIO_BUTTON,
        )
        train_options_gb.layout().addWidget(self._use_cudnn_bench)

        model_gb = GroupBox('Model options')
        layout.addWidget(model_gb)

        self._lr = Parameter('lr', [0.0004])
        model_gb.layout().addWidget(self._lr)

        self._gdeep = Parameter(
            'gdeep',
            [True, False],
            True,
            WIDGET_TYPE.RADIO_BUTTON,
        )
        model_gb.layout().addWidget(self._gdeep)

        self._beta1 = Parameter('beta 1', [0])
        model_gb.layout().addWidget(self._beta1)

        self._lambda_id = Parameter('lambda id', [30])
        model_gb.layout().addWidget(self._lambda_id)

        self._lambda_feat = Parameter('lambda_feat', [10])
        model_gb.layout().addWidget(self._lambda_feat)

        self._lambda_rec = Parameter('lambda_rec', [10])
        model_gb.layout().addWidget(self._lambda_rec)

        dataset_gb = GroupBox('Dataset options')
        layout.addWidget(dataset_gb)

        self._dataset_path_row = SelectDirRow(
            'Dataset',
            # r'C:\Users\tonkec\Desktop\vggface2_crop_arcfacealign_224',
            r'C:\Users\tonkec\Desktop\trump_cage_dataset',
        )
        dataset_gb.layout().addWidget(self._dataset_path_row)

        self._bs = Parameter('batch size', [32])
        dataset_gb.layout().addWidget(self._bs)

        self._log_config_wgt = LoggingConfig(FREQUENCY_UNIT.STEP)
        layout.addWidget(self._log_config_wgt)

        desc_gb = GroupBox('Description')
        layout.addWidget(desc_gb)

        self._desc_input = qwt.QTextEdit()
        desc_gb.layout().addWidget(self._desc_input)

        self._update_runs_selection()

        layout.addItem(VerticalSpacer())

    @property
    def batch_size(self) -> int:
        return int(self._bs.value)

    @property
    def gdeep(self) -> bool:
        return str_to_bool(self._gdeep.value)

    @property
    def steps(self) -> int:
        return int(self._steps.value)

    @property
    def lr(self) -> float:
        return float(self._lr.value)

    @property
    def beta1(self) -> float:
        return float(self._beta1.value)

    @property
    def lambda_id(self) -> float:
        return float(self._lambda_id.value)

    @property
    def lambda_feat(self) -> float:
        return float(self._lambda_feat.value)

    @property
    def lambda_rec(self) -> float:
        return float(self._lambda_rec.value)

    @property
    def use_cudnn(self) -> bool:
        return str_to_bool(self._use_cudnn_bench.value)

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_path_row.selected_dir

    @property
    def resume(self) -> bool:
        return str_to_bool(self._resume.value)

    @property
    def resume_run_name(self) -> Optional[str]:
        val = self._run._cb.currentText()
        if val:
            return val
        return None

    @property
    def log_frequency(self) -> int:
        return self._log_config_wgt.log_frequency

    @property
    def sample_frequency(self) -> int:
        return self._log_config_wgt.sample_frequency

    @property
    def checkpoint_frequency(self) -> int:
        return self._log_config_wgt.checkpoint_frequency

    @property
    def run_description(self) -> str:
        return self._desc_input.toPlainText()

    @qtc.pyqtSlot(int)
    def _run_selection_changed(self, index: int) -> None:
        self.run_changed_sig.emit()

    def _update_runs_selection(self) -> None:
        runs_dir = self._log_config_wgt.log_dir / self._model_name
        runs = [str(p.stem) for p in list(runs_dir.glob('*'))]
        self._run._cb.addItems(reversed(runs))
        if len(runs):
            self.enable_widget(self._run_info_btn, True)
        else:
            self.enable_widget(self._run_info_btn, False)
        logger.debug(
            f'Refreshed list of runs. Found total of {len(runs)} runs.'
        )

    @qtc.pyqtSlot()
    def _refresh_runs(self) -> None:
        self._run._cb.clear()
        self._update_runs_selection()

    @qtc.pyqtSlot()
    def _run_info(self) -> None:
        print('inf')