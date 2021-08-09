import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER

from message.message import Message


class SignalProvider(qtc.QObject):
    __instance = None
    __signals = {}

    @staticmethod
    def get_instance():
        if SignalProvider.__instance is None:
            SignalProvider()
        return SignalProvider.__instance

    @staticmethod
    def get_signal(signal_owner: SIGNAL_OWNER):
        return SignalProvider.get_instance().__signals[signal_owner]

    def __init__(self):
        if SignalProvider.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SignalProvider.__instance = self
            SignalProvider.__signals = {
                # workers
                SIGNAL_OWNER.IO_WORKER: qtc.pyqtSignal(Message),
                SIGNAL_OWNER.MESSAGE_WORKER: qtc.pyqtSignal(Message),
                SIGNAL_OWNER.FACE_DETECTION_WORKER: qtc.pyqtSignal(Message),
                SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER: qtc.pyqtSignal(Message),
                SIGNAL_OWNER.MESSAGE_WORKER: qtc.pyqtSignal(Message),
                # widget actions
                SIGNAL_OWNER.CONFIGURE_WIDGET: qtc.pyqtSignal(Message),
                SIGNAL_OWNER.JOB_PROGRESS: qtc.pyqtSignal(Message),
            }
