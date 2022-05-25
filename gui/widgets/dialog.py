import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt
import PyQt6.QtSvgWidgets as qts

from common_structures import DialogMessages
from gui.widgets.common import NoMarginLayout


class Dialog(qwt.QDialog):

    remove_sig = qtc.pyqtSignal(bool)

    def __init__(self, dialog_msg: DialogMessages, parent=None):
        super().__init__(parent)

        self.setWindowTitle(dialog_msg.message_type)

        QBtn = qwt.QDialogButtonBox.StandardButton.Ok | \
            qwt.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = qwt.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accepted)
        self.buttonBox.rejected.connect(self.rejected)

        self.layout = qwt.QVBoxLayout()
        self.message = qwt.QLabel(dialog_msg.message)
        self.icon_message_layout = qwt.QHBoxLayout()
        svg_trash = qts.QSvgWidget(dialog_msg.message_icon)
        svg_trash.setFixedSize(svg_trash.width() / 4, svg_trash.height() / 4)
        self.icon_message_layout.addWidget(svg_trash)
        self.icon_message_layout.addWidget(self.message)
        self.layout.addLayout(self.icon_message_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def accepted(self) -> None:
        self.remove_sig.emit(True)
        self.close()

    def rejected(self) -> None:
        self.remove_sig.emit(False)
        self.close()


class InfoDialog(qwt.QDialog):

    def __init__(self, title: str = '', message: str = '') -> None:
        super().__init__()

        self._message = message

        self.setWindowTitle(title)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        layout.addStretch()
        layout.addWidget(qwt.QLabel(text=self._message))
        layout.addStretch()
