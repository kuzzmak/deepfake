import time

import PyQt5.QtCore as qtc


class Worker(qtc.QObject):

    incremented_val = qtc.pyqtSignal(int)

    @qtc.pyqtSlot(int)
    def increment_value(self, value: int):
        new_value = value
        while new_value < 100:
            new_value += 1
            time.sleep(0.05)
            self.incremented_val.emit(new_value)
