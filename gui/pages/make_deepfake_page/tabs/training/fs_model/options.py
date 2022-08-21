import PyQt6.QtWidgets as qwt


class Options(qwt.QWidget):

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(qwt.QPushButton(text='heeelo')) 