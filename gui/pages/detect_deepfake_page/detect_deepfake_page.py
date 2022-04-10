from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
from config import APP_CONFIG

from enums import SIGNAL_OWNER
from gui.pages.detect_deepfake_page.meso_net_widget import MesoNetWidget
from gui.pages.detect_deepfake_page.mri_gan_widget import MriGanWidget
from gui.pages.page import Page
from gui.widgets.common import HWidget, HorizontalSpacer, VWidget, VerticalSpacer
from names import DETECT_DEEPFAKE_PAGE_NAME


class DetectDeepFakePage(Page):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals, DETECT_DEEPFAKE_PAGE_NAME)

        self._init_ui()

    def _init_ui(self) -> None:
        _layout = qwt.QVBoxLayout()
        self.setLayout(_layout)
        central_wgt = qwt.QWidget()
        _layout.addWidget(central_wgt)
        central_wgt_layout = qwt.QVBoxLayout()
        central_wgt.setLayout(central_wgt_layout)

        model_selection_row = HWidget()
        central_wgt_layout.addWidget(model_selection_row)
        model_selection_row.layout().setContentsMargins(0, 0, 0, 0)
        model_selection_row.layout().addWidget(qwt.QLabel(
            text='Model selection'
        ))
        self.model_cb = qwt.QComboBox()
        model_selection_row.layout().addWidget(self.model_cb)
        model_selection_row.layout().addItem(HorizontalSpacer)
        self.model_cb.currentIndexChanged.connect(
            self._model_selection_changed
        )
        self.model_cb.setMaximumWidth(150)
        for model in APP_CONFIG.app.core.df_detection.models:
            self.model_cb.addItem(model.name, model.id)

        self.stacked_wgt = qwt.QStackedWidget()
        central_wgt_layout.addWidget(self.stacked_wgt)

        self.meso_net_wgt = MesoNetWidget()
        self.stacked_wgt.addWidget(self.meso_net_wgt)

        self.mri_gan_wgt = MriGanWidget()
        self.stacked_wgt.addWidget(self.mri_gan_wgt)

    def _model_selection_changed(self, idx: int) -> None:
        """Triggers when model selection is changed.

        Args:
            idx (int): index of the item that is selected
        """
        model_id = self.model_cb.currentData()
        model_conf = list(filter(
            lambda model: model.id == model_id,
            APP_CONFIG.app.core.df_detection.models
        ))
