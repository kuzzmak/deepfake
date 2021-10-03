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

MinimalSizePolicy = qwt.QSizePolicy(
    qwt.QSizePolicy.Fixed,
    qwt.QSizePolicy.Fixed,
)
