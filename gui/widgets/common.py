from typing import Any, List, Union

import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.app_config import APP_CONFIG
from enums import DEVICE, LAYOUT


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
