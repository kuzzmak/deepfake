from gui.workers.worker import Worker

from message.message import Message


class MessageWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        self.signals[msg.recipient].emit(msg)
