# import time
import abc
import PyQt5
import PyQt5.QtCore as qtc

from message.message import Message

from enums import SIGNAL_OWNER


class WorkerMeta(PyQt5.sip.wrappertype, abc.ABCMeta):
    ...


class Worker(qtc.QObject, metaclass=WorkerMeta):

    signals = dict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_signal(self, sig: qtc.pyqtSignal, sig_owner: SIGNAL_OWNER):
        self.signals[sig_owner] = sig

    @abc.abstractclassmethod
    @qtc.pyqtSlot(Message)
    def process(self, msg: Message):
        ...
