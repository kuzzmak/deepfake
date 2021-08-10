import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt


class JobInfoWindow(qwt.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Job info')
        self.resize(500, 250)

        central_wgt = qwt.QWidget(self)
        central_wgt_layout = qwt.QVBoxLayout()
        central_wgt.setLayout(central_wgt_layout)

        hello_btn = qwt.QPushButton(text='Press me')
        hello_btn.clicked.connect(self.hello)
        central_wgt_layout.addWidget(hello_btn)

        self.setCentralWidget(central_wgt)

    @qtc.pyqtSlot()
    def hello(self):
        print('hello')
