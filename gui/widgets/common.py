import PyQt5.QtWidgets as qwt


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
