import sys
import PyQt5

from PyQt5.QtGui import QFont

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget

from constants import MAX_WIDTH, MAX_HEIGHT


class Ui(QWidget):

    def __init__(self):
        super(Ui, self).__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout.setSpacing(20)

        font = QFont('Times', 15)

        label = QLabel('Deepfake')
        label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        label.setFont(font)
        label.setFixedSize(250, 200)
        layout.addWidget(label)

        make_deepfake_btn = QPushButton('Make deepfake')
        make_deepfake_btn.setFixedSize(250, 50)
        make_deepfake_btn.setFont(font)
        layout.addWidget(make_deepfake_btn)

        detect_deepfake_btn = QPushButton('Detect deepfake')
        detect_deepfake_btn.setFont(font)
        detect_deepfake_btn.setFixedSize(250, 50)
        layout.addWidget(detect_deepfake_btn)

        self.setLayout(layout)

        self.setGeometry(0, 0, MAX_WIDTH, MAX_HEIGHT)
        self.setMaximumHeight(MAX_HEIGHT)
        self.setMaximumWidth(MAX_WIDTH)
        self.setWindowTitle('Deepfake')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()
    sys.exit(app.exec_())
