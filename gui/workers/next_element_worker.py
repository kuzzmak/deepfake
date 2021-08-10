from enums import BODY_KEY, SIGNAL_OWNER

from gui.workers.worker import Worker

from message.message import Message


class NextElementWorker(Worker):

    def __init__(self, signals, *args, **kwargs):
        super().__init__(signals, *args, **kwargs)

    def process(self, msg: Message):
        data = msg.body.data

        worker = data[BODY_KEY.SIGNAL_OWNER]

        if worker == SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER:
            self.signals[SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER_NEXT_ELEMENT].emit()
