import PyQt5.QtWidgets as qwt

from enums import LAYOUT


VerticalSpacer = qwt.QSpacerItem(
    1,
    1,
    qwt.QSizePolicy.Fixed,
    qwt.QSizePolicy.MinimumExpanding,
)

HorizontalSpacer = qwt.QSpacerItem(
    1,
    1,
    qwt.QSizePolicy.MinimumExpanding,
    qwt.QSizePolicy.Fixed,
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


MinimalSizePolicy = qwt.QSizePolicy(
    qwt.QSizePolicy.Fixed,
    qwt.QSizePolicy.Fixed,
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
        qwt.QStyle.SP_MessageBoxInformation
    ))
    button.setStyleSheet("border-radius: 10")
    button.setToolTip(tooltip)
    return button
