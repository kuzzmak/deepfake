from enums import BODY_KEY, JOB_TYPE, SIGNAL_OWNER
from gui.workers.worker import Worker

from message.message import Message


class MessageWorker(Worker):

    def __init__(self, signals, *args, **kwargs):
        super().__init__(signals, *args, **kwargs)

    def process(self, msg: Message):

        if msg.recipient == SIGNAL_OWNER.MESSAGE_WORKER:
            if msg.body.job_type == JOB_TYPE.ADD_SIGNAL:
                data = msg.body.data
                signal_owner = data[BODY_KEY.SIGNAL_OWNER]
                signal = data[BODY_KEY.SIGNAL]
                self.signals[signal_owner] = signal
        else:
            self.signals[msg.recipient].emit(msg)
