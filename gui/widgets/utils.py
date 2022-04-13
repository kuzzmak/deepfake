import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt


def set_color_on_widget(
    widget: qwt.QWidget,
    color: qtc.Qt.GlobalColor = qtc.Qt.red,
):
    """Sets some color on the widget. Can be used for styling or debugging
    purposes where true widget size can be seen only when color is applied.

    Args:
        widget (qwt.QWidget): widget which color is about to change
        color (qtc.Qt.GlobalColor, optional): paint widget to which color.
            Defaults to qtc.Qt.red.
    """
    widget.setAutoFillBackground(True)
    p = widget.palette()
    p.setColor(widget.backgroundRole(), color)
    widget.setPalette(p)
