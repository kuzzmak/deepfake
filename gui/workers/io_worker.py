import os

import PyQt5.QtCore as qtc

from enums import IO_OP_TYPE
from common_structures import IO_OP


class IO_Worker(qtc.QObject):

    # incremented_val = qtc.pyqtSignal(int)
    io_op_successful_sig = qtc.pyqtSignal(bool)

    @qtc.pyqtSlot(IO_OP)
    def io_op(self, op: IO_OP):
        if op.type == IO_OP_TYPE.DELETE:
            self.delete(op.value)

    def delete(self, value: str):
        exists = os.path.exists(value)
        if exists:
            os.remove(value)
            print('removed: ', value)

    # @qtc.pyqtSlot(int)
    # def increment_value(self, value: int):
    #     new_value = value
    #     while new_value < 100:
    #         new_value += 1
    #         time.sleep(0.05)
    #         self.incremented_val.emit(new_value)
