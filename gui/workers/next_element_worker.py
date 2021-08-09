from enums import BODY_KEY

from gui.workers.worker import Worker

from message.message import Message


class NextElementWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        data = msg.body.data

        print(msg)

        worker = data[BODY_KEY.SIGNAL_OWNER]

        # print(self.signals)
        self.signals[worker].emit(True)
