import PyQt6.QtGui as qtg
import PyQt6.QtWidgets as qwt

from enums import LAYOUT


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
