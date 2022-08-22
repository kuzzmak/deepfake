import PyQt6.QtWidgets as qwt

from enums import FREQUENCY_UNIT, WIDGET_TYPE
from gui.pages.make_deepfake_page.tabs.training.widgets import LoggingConfig
from gui.widgets.common import GroupBox, Parameter, VerticalSpacer


class Options(qwt.QWidget):

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        model_gb = GroupBox('Model options')
        layout.addWidget(model_gb)

        self._bs = Parameter('batch size', [1])
        model_gb.layout().addWidget(self._bs)

        self._steps = Parameter('steps', [100])
        model_gb.layout().addWidget(self._steps)

        self._lr = Parameter('lr', [0.0004])
        model_gb.layout().addWidget(self._lr)

        self._gdeep = Parameter(
            'gdeep',
            [True, False],
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

        self._use_cudnn_bench = Parameter(
            'use cudnn benchmark',
            [True, False],
            WIDGET_TYPE.RADIO_BUTTON,
        )
        model_gb.layout().addWidget(self._use_cudnn_bench)

        self._log_config_wgt = LoggingConfig(FREQUENCY_UNIT.STEP)
        layout.addWidget(self._log_config_wgt)

        layout.addItem(VerticalSpacer())
