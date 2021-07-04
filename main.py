from PyQt5.QtWidgets import QApplication
from gui.gui import App


if __name__ == '__main__':
    _app = QApplication([])
    app = App()
    app.show()
    _app.exec_()
