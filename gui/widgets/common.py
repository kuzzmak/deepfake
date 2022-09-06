from typing import Any, List, Optional, Union

import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.app_config import APP_CONFIG
from enums import DEVICE, LAYOUT, WIDGET_TYPE


def VerticalSpacer() -> qwt.QSpacerItem:
    return qwt.QSpacerItem(
        1,
        1,
        qwt.QSizePolicy.Policy.Fixed,
        qwt.QSizePolicy.Policy.MinimumExpanding,
    )


def HorizontalSpacer() -> qwt.QSpacerItem:
    return qwt.QSpacerItem(
        1,
        1,
        qwt.QSizePolicy.Policy.MinimumExpanding,
        qwt.QSizePolicy.Policy.Fixed,
    )


def HWidget() -> qwt.QWidget:
    wgt = qwt.QWidget()
    wgt_layout = qwt.QHBoxLayout()
    wgt.setLayout(wgt_layout)
    return wgt


def VWidget() -> qwt.QWidget:
    wgt = qwt.QWidget()
    wgt_layout = qwt.QVBoxLayout()
    wgt.setLayout(wgt_layout)
    return wgt


def MinimalSizePolicy() -> qwt.QSizePolicy:
    return qwt.QSizePolicy(
        qwt.QSizePolicy.Policy.Fixed,
        qwt.QSizePolicy.Policy.Fixed,
    )


def GroupBox(
    title: str = '',
    layout: LAYOUT = LAYOUT.VERTICAL,
) -> qwt.QGroupBox:
    gb = qwt.QGroupBox()
    gb.setLayout(
        qwt.QVBoxLayout()
        if layout == LAYOUT.VERTICAL
        else qwt.QHBoxLayout()
    )
    gb.setTitle(title)
    return gb


class Button(qwt.QPushButton):

    def __init__(
        self,
        text: str = 'click',
        tooltip: str = '',
        width: int = 150,
    ):
        super().__init__(text)

        self.setToolTip(tooltip)
        self.setFixedWidth(width)


class IconButton(Button):

    def __init__(
        self,
        icon: qtg.QIcon,
        tooltip: str = '',
    ) -> None:
        super().__init__('', tooltip, 20)

        self.setIcon(icon)
        self.setStyleSheet("border-radius: 10")


def InfoIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_MessageBoxInformation
    )


def InfoButton(tooltip: str = '') -> IconButton:
    return IconButton(InfoIcon(), tooltip)


def PlayIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_MediaPlay
    )


def StopIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_MediaStop
    )


def ApplyIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_DialogApplyButton
    )


def CancelIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_DialogCancelButton
    )


def ApplyIconButton() -> IconButton:
    return IconButton(ApplyIcon())


def CancelIconButton() -> IconButton:
    return IconButton(CancelIcon())


def NoMarginLayout(
    layout: LAYOUT = LAYOUT.VERTICAL
) -> Union[qwt.QHBoxLayout, qwt.QVBoxLayout]:
    """Used for constructing widget layout based on the type of the layout
    `layout` which has no margins.

    Parameters
    ----------
    layout : LAYOUT, optional
        layout type, by default LAYOUT.VERTICAL

    Returns
    -------
    Union[qwt.QHBoxLayout, qwt.QVBoxLayout]
        layout with no margins based on thy type of the argument
    """
    if layout == LAYOUT.VERTICAL:
        _lay = qwt.QVBoxLayout()
    else:
        _lay = qwt.QHBoxLayout()
    _lay.setContentsMargins(0, 0, 0, 0)
    return _lay


class DeviceWidget(qwt.QWidget):
    """Widget containing available devices on which some process could run.
    """

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        gb = GroupBox('Device', LAYOUT.HORIZONTAL)
        self._device_bg = qwt.QButtonGroup(gb)
        layout.addWidget(gb)
        for device in APP_CONFIG.app.core.devices:
            btn = qwt.QRadioButton(device.value, gb)
            btn.setChecked(True)
            self._device_bg.addButton(btn)
            gb.layout().addWidget(btn)

    @ property
    def device(self) -> DEVICE:
        """Currently selected device.

        Returns:
            DEVICE: cpu or cuda
        """
        for but in self._device_bg.buttons():
            if but.isChecked():
                return DEVICE[but.text().upper()]
        return DEVICE.CPU


class DeviceRow(qwt.QWidget):
    """Simple widget containing available devices in a row.
    """

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout(LAYOUT.HORIZONTAL)
        self.setLayout(layout)
        layout.addWidget(qwt.QLabel(text='device'))
        self._device_bg = qwt.QButtonGroup(self)
        for device in APP_CONFIG.app.core.devices:
            btn = qwt.QRadioButton(device.value)
            btn.setChecked(True)
            self._device_bg.addButton(btn)
            layout.addWidget(btn)

    @property
    def device(self) -> DEVICE:
        """Currently selected device.

        Returns:
            DEVICE: cpu or cuda
        """
        for but in self._device_bg.buttons():
            if but.isChecked():
                return DEVICE[but.text().upper()]
        return DEVICE.CPU


class RadioButtons(qwt.QWidget):
    """Widget which consists of radio buttons.

    Parameters
    ----------
    options : List[str]
        list of names for each radio button
    layout : LAYOUT, optional
        how is this widget orientated, by default LAYOUT.HORIZONTAL
    """

    selection_changed_sig = qtc.pyqtSignal(list)

    def __init__(
        self,
        options: List[str],
        layout: LAYOUT = LAYOUT.HORIZONTAL,
    ) -> None:
        super().__init__()

        self._options = options
        self._layout_type = layout

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout(self._layout_type)
        self.setLayout(layout)
        self._bg = qwt.QButtonGroup(self)
        for opt in self._options:
            btn = qwt.QRadioButton(opt)
            self._bg.addButton(btn)
            layout.addWidget(btn)
        if len(self._options) == 0:
            raise Exception('At least one options needs to be present.')
        self._bg.buttons()[0].setChecked(True)
        self._bg.idPressed.connect(self._id_changed)

    @property
    def selected(self) -> Any:
        for but in self._bg.buttons():
            if but.isChecked():
                return but.text()

    @qtc.pyqtSlot(int)
    def _id_changed(self, id: int) -> None:
        self.selection_changed_sig.emit([self._options[abs(-2 - id)]])


class Parameter(qwt.QWidget):
    """Widget for showing input or selection for some model parameter.

    Parameters
    ----------
    name : str
        name of the parameter in GUI
    default_values : Optional[List[Any]], optional
        put default values in input, by default None
    widget_type : WIDGET_TYPE, optional
        is this widget input widget or radio buttons, by default
            WIDGET_TYPE.INPUT
    """

    def __init__(
        self,
        name: str,
        values: Optional[List[Any]] = None,
        default_value: Optional[Any] = None,
        widget_type: WIDGET_TYPE = WIDGET_TYPE.INPUT,
    ) -> None:
        super().__init__()

        self._name = name
        self._values = values
        self._default_value = default_value
        self._wt = widget_type

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout(LAYOUT.HORIZONTAL)
        self.setLayout(layout)
        if self._name:
            layout.addWidget(qwt.QLabel(text=self._name))
            layout.addItem(HorizontalSpacer())

        if self._wt == WIDGET_TYPE.INPUT:
            self._input = qwt.QLineEdit()
            self._input.setMaximumWidth(100)
            layout.addWidget(self._input)
            if self._default_value is not None:
                self._input.setText(str(self._default_value))
            else:
                if self._values is None:
                    return
                self._input.setText(str(self._values[0]))

        elif self._wt == WIDGET_TYPE.RADIO_BUTTON:
            if self._values is None:
                return
            vals = self._values
            if len(vals) == 0:
                return
            self._btn_bg = qwt.QButtonGroup(self)
            for idx, val in enumerate(vals):
                btn = qwt.QRadioButton(str(val))
                if idx == 0:
                    btn.setChecked(True)
                if self._default_value is not None and \
                        val == self._default_value:
                    btn.setChecked(True)
                self._btn_bg.addButton(btn)
                layout.addWidget(btn)

        elif self._wt == WIDGET_TYPE.DROPDOWN:
            self._cb = qwt.QComboBox()
            layout.addWidget(self._cb)
            self._cb.addItems(self._values)

    @property
    def value(self) -> Any:
        if self._wt == WIDGET_TYPE.INPUT:
            return self._input.text()
        elif self._wt == WIDGET_TYPE.RADIO_BUTTON:
            for but in self._btn_bg.buttons():
                if but.isChecked():
                    return but.text()
        elif self._wt == WIDGET_TYPE.DROPDOWN:
            return self._cb.currentText()
