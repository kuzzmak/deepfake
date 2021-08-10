from typing import Dict, Optional

import PyQt5.QtCore as qtc

import cv2 as cv

from enums import (
    BODY_KEY,
    FILE_TYPE,
    IO_OPERATION_TYPE,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)

from gui.workers.worker import Worker

from message.message import Body, Message

from utils import resize_image_retain_aspect_ratio


class IO_Worker(Worker):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        *args,
        **kwargs
    ):
        super().__init__(signals, *args, **kwargs)

    def process(self, msg: Message):
        data = msg.body.data
        io_operation_type = data[BODY_KEY.IO_OPERATION_TYPE]
        file_path = data[BODY_KEY.FILE_PATH]
        file_type = data[BODY_KEY.FILE_TYPE]

        if io_operation_type == IO_OPERATION_TYPE.SAVE:

            file = data[BODY_KEY.FILE]

            if file_type == FILE_TYPE.IMAGE:

                resize = data[BODY_KEY.RESIZE]

                if resize:
                    new_size = data[BODY_KEY.NEW_SIZE]
                    file = resize_image_retain_aspect_ratio(file, new_size)

                cv.imwrite(file_path, file)

        multipart = data[BODY_KEY.MULTIPART]
        if multipart:
            part = data[BODY_KEY.PART]
            total = data[BODY_KEY.TOTAL]

            msg = Message(
                MESSAGE_TYPE.ANSWER,
                MESSAGE_STATUS.OK,
                SIGNAL_OWNER.IO_WORKER,
                SIGNAL_OWNER.JOB_PROGRESS,
                Body(
                    JOB_TYPE.IO_OPERATION,
                    {
                        BODY_KEY.PART: part,
                        BODY_KEY.TOTAL: total,
                    },
                    part == total,
                )
            )
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

        # message for next frame
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.IO_WORKER,
            SIGNAL_OWNER.NEXT_ELEMENT_WORKER,
            Body(
                JOB_TYPE.NEXT_ELEMENT,
                {
                    BODY_KEY.SIGNAL_OWNER:
                    SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
                }
            )
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)
