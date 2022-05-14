from typing import Union
import PyQt6.QtGui as qtg
import PyQt6.QtWidgets as qwt

from config import APP_CONFIG
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


def Button(text: str = 'click', width: int = 150) -> qwt.QPushButton:
    but = qwt.QPushButton(text=text)
    but.setFixedWidth(width)
    return but


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


def InfoButton(tooltip: str = '') -> Button:
    button = Button('', 20)
    button.setIcon(qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_MessageBoxInformation
    ))
    button.setStyleSheet("border-radius: 10")
    button.setToolTip(tooltip)
    return button


def PlayIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_MediaPlay
    )


def StopIcon() -> qtg.QIcon:
    return qwt.QApplication.style().standardIcon(
        qwt.QStyle.StandardPixmap.SP_MediaStop
    )


def NoMarginLayout(
    layout: LAYOUT = LAYOUT.VERTICAL,
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
