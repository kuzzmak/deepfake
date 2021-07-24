from enums import SIGNAL_OWNER
from gui.workers.worker import Worker

from message.message import JOB_TYPE, MESSAGE_TYPE, Message


class MessageWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        if msg.type == MESSAGE_TYPE.REQUEST:

            job_type = msg.body.job_type

            if job_type == JOB_TYPE.CONSOLE_PRINT:
                self.signals[SIGNAL_OWNER.CONOSLE].emit(msg)

            elif job_type == JOB_TYPE.FRAME_EXTRACTION:
                self.signals[SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER].emit(msg)

            elif job_type == JOB_TYPE.IO_OPERATION:
                self.signals[SIGNAL_OWNER.IO_WORKER].emit(msg)
