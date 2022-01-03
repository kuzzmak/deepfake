import logging
import os
from typing import Any, Dict, Optional

import PyQt5.QtCore as qtc
import cv2 as cv

from enums import (
    BODY_KEY,
    DATA_TYPE,
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

logger = logging.getLogger(__name__)


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
        if io_operation_type == IO_OPERATION_TYPE.SAVE:
            self._handle_save_op(msg.sender, data)
        elif io_operation_type == IO_OPERATION_TYPE.DELETE:
            self._handle_delete_op(data)

    def _handle_delete_op(self, data: Dict[BODY_KEY, Any]) -> None:
        file_path = data[BODY_KEY.FILE_PATH]
        try:
            os.remove(file_path)
            logger.debug(f'Removed file {file_path}.')
        except FileNotFoundError:
            logger.warning(
                f'Unable to remove file: {file_path} because it\'s in use.'
            )
        except Exception:
            logger.error(
                f'Error happened while trying to remove file {file_path}.'
            )

    def _handle_save_op(
        self,
        sender: SIGNAL_OWNER,
        data: Dict[BODY_KEY, Any],
    ) -> None:
        file_path = data[BODY_KEY.FILE_PATH]
        file_type = data[BODY_KEY.FILE_TYPE]
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
            _msg = Message(
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
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(_msg)

        # message for next frame
        if sender == SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER:
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
