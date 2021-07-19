import os

import PyQt5.QtCore as qtc

from enums import IO_OP_TYPE
from common_structures import IO_OP


class IO_Worker(qtc.QObject):

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
