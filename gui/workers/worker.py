# import time
import abc
import PyQt5

import PyQt5.QtCore as qtc

# class Worker(qtc.QObject):

#     incremented_val = qtc.pyqtSignal(int)

#     @qtc.pyqtSlot(int)
#     def increment_value(self, value: int):
#         new_value = value
#         while new_value < 100:
#             new_value += 1
#             time.sleep(0.05)
#             self.incremented_val.emit(new_value)


class WorkerMeta(PyQt5.sip.wrappertype, abc.ABCMeta):
    ...


class Worker(qtc.QObject, metaclass=WorkerMeta):

    succesful_op = qtc.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.succesful_op.connect(self.print)

    @qtc.pyqtSlot(bool)
    def print(self, val: bool):
        print('val: ', val)

    @abc.abstractclassmethod
    def process(self, data):
        ...


class IncrementerWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, data):
        print('data from inc worker: ', data)
        self.succesful_op.emit(False)


if __name__ == '__main__':

    wor = IncrementerWorker()
    wor.process('s')
