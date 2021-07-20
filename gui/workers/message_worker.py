from gui.workers.worker import NewWorker
from message.message import Message


class MessageWorker(NewWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        op, file = msg.get_data()
        print(op)
        print(file)
