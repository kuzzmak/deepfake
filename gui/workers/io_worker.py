import cv2 as cv

from message.message import Message

from gui.workers.worker import Worker

from enums import IO_OP_TYPE


class IO_Worker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):

        op_type, file_path, new_file_path, file = msg.body.get_data()

        if op_type == IO_OP_TYPE.SAVE:
            if file is not None:
                cv.imwrite(file_path, file)
