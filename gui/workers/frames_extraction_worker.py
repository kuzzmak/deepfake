from message.message import Message

from gui.workers.worker import Worker


class FramesExtractionWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(msg: Message):
        print('heeelo')
