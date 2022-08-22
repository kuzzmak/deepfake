from pathlib import Path

import PyQt6.QtWidgets as qwt

from enums import FREQUENCY_UNIT, LAYOUT, WIDGET_TYPE
from gui.widgets.common import (
    Button,
    GroupBox,
    HorizontalSpacer,
    HWidget,
    NoMarginLayout,
    Parameter,
)


class SelectDirRow(qwt.QWidget):

    def __init__(self, name: str, default_value: str = '') -> None:
        super().__init__()

        self._name = name
        self._default_value = default_value

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        first_row = HWidget()
        layout.addWidget(first_row)
        first_row.layout().setContentsMargins(0, 0, 0, 0)
        dir_lbl = qwt.QLabel(text=f'{self._name} directory')
        first_row.layout().addWidget(dir_lbl)
        first_row.layout().addItem(HorizontalSpacer())
        self._select_dir_btn = Button('select', width=100)
        first_row.layout().addWidget(self._select_dir_btn)

        second_row = HWidget()
        layout.addWidget(second_row)
        second_row.layout().setContentsMargins(0, 0, 0, 0)
        dir_path = Path(self._default_value).absolute()
        self._selected_dir_lbl = qwt.QLabel(text=str(dir_path))
        second_row.layout().addWidget(self._selected_dir_lbl)
        self._selected_dir_lbl.setToolTip(str(dir_path))
        self._selected_dir_lbl.setMaximumWidth(350)
        second_row.layout().addItem(HorizontalSpacer())

    @property
    def selected_dir(self) -> Path:
        return Path(self._selected_dir_lbl.text())


class FrequencyRow(qwt.QWidget):

    def __init__(self, name: str, default_value: int,
                 frequency_unit: FREQUENCY_UNIT) -> None:
        super().__init__()

        self._name = name
        self._default_value = default_value
        self._frequency_unit = frequency_unit

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout(LAYOUT.HORIZONTAL)
        self.setLayout(layout)
        layout.addWidget(qwt.QLabel(text=f'{self._name} frequency'))
        layout.addItem(HorizontalSpacer())
        self._freq_input = qwt.QLineEdit(text=str(self._default_value))
        layout.addWidget(self._freq_input)
        self._freq_input.setMaximumWidth(100)
        layout.addWidget(qwt.QLabel(text=self._frequency_unit.value))

    @property
    def frequency(self) -> int:
        return int(self._freq_input.text())


class LoggingConfig(qwt.QWidget):

    def __init__(self, frequency_unit: FREQUENCY_UNIT) -> None:
        super().__init__()

        self._frequency_unit = frequency_unit

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        self._gb = GroupBox('Logging config')
        layout.addWidget(self._gb)
        self._gb.layout().setSpacing(20)

        self._log_dir = SelectDirRow('Logging', 'logs')
        self._gb.layout().addWidget(self._log_dir)

        self._checkpoints_dir = SelectDirRow('Checkpoints', 'checkpoints')
        self._gb.layout().addWidget(self._checkpoints_dir)

        logg_freq_row = FrequencyRow('Logging', 200, self._frequency_unit)
        self._gb.layout().addWidget(logg_freq_row)

        chkpt_freq_row = FrequencyRow('Checkpoint', 500, self._frequency_unit)
        self._gb.layout().addWidget(chkpt_freq_row)

        sample_freq_row = FrequencyRow('Sample', 500, self._frequency_unit)
        self._gb.layout().addWidget(sample_freq_row)

        self._use_wandb = Parameter(
            'use wandb',
            [True, False],
            WIDGET_TYPE.RADIO_BUTTON,
        )
        self._gb.layout().addWidget(self._use_wandb)
