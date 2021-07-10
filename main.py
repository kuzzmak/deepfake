import sys

import PyQt5.QtWidgets as qwt
from gui.gui import App


if __name__ == '__main__':
    _app = qwt.QApplication(sys.argv)
    app = App()
    app.show()
    sys.exit(_app.exec_())
