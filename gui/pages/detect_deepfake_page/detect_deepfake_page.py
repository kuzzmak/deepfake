from typing import Dict, Optional

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.app_config import APP_CONFIG
from enums import DF_DETECTION_MODEL, SIGNAL_OWNER
from gui.pages.detect_deepfake_page.meso_net_widget import MesoNetWidget
from gui.pages.detect_deepfake_page.mri_gan.mri_gan_widget import MRIGANWidget
from gui.pages.page import Page
from gui.widgets.common import HWidget, HorizontalSpacer, NoMarginLayout
from variables import DETECT_DEEPFAKE_PAGE_NAME


class DetectDeepFakePage(Page):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals, DETECT_DEEPFAKE_PAGE_NAME)

        self._init_ui()

    def _init_ui(self) -> None:
        _layout = NoMarginLayout()
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
        model_selection_row.layout().addItem(HorizontalSpacer())
        self.model_cb.currentIndexChanged.connect(
            self._model_selection_changed
        )
        self.model_cb.setMaximumWidth(150)

        mri_gan = APP_CONFIG.app.core.df_detection.models.mri_gan
        self.model_cb.addItem(mri_gan.name, mri_gan.id)
        meso_net = APP_CONFIG.app.core.df_detection.models.meso_net
        self.model_cb.addItem(meso_net.name, meso_net.id)
        self.model_cb.setCurrentIndex(0)

        self.stacked_wgt = qwt.QStackedWidget()
        central_wgt_layout.addWidget(self.stacked_wgt)

        message_worker_sig = self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
        signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: message_worker_sig
        }

        self.meso_net_wgt = MesoNetWidget(signals)
        self.stacked_wgt.addWidget(self.meso_net_wgt)

        self.mri_gan_wgt = MRIGANWidget(signals)
        self.stacked_wgt.addWidget(self.mri_gan_wgt)

        self.stacked_wgt.setCurrentWidget(self.mri_gan_wgt)

    def _model_selection_changed(self, idx: int) -> None:
        """Triggers when model selection is changed.

        Args:
            idx (int): index of the item that is selected
        """
        try:
            model_id = self.model_cb.currentData()
            wgt_to_set = self.meso_net_wgt \
                if model_id == DF_DETECTION_MODEL.MESO_NET.value \
                else self.mri_gan_wgt
            self.stacked_wgt.setCurrentWidget(wgt_to_set)
        except AttributeError:
            # triggers before ui is initiated
            ...
