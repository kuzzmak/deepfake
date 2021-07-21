from gui.workers.worker import NewWorker
from message.message import JOB_TYPE, MESSAGE_TYPE, Message


class MessageWorker(NewWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        if msg.type == MESSAGE_TYPE.REQUEST:
            if msg.body.job_type == JOB_TYPE.CONSOLE_PRINT:
                msg_type, message = msg.body.get_data()
                print('mes type: ', msg_type)
                print('messa: ', message)
