import cv2 as cv

from message.message import AnswerBody, Message

from gui.workers.worker import Worker

from enums import IO_OP_TYPE, MESSAGE_STATUS, MESSAGE_TYPE, SIGNAL_OWNER


class IO_Worker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):

        op_type, file_path, new_file_path, file, multipart, part, total = msg.body.get_data()

        if op_type == IO_OP_TYPE.SAVE:
            if file is not None:
                cv.imwrite(file_path, file)
                if multipart:
                    msg = Message(
                        MESSAGE_TYPE.ANSWER,
                        AnswerBody(
                            MESSAGE_STATUS.OK,
                            part == total,
                        )
                    )
                    self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)
                else:
                    ...
