import os
from typing import Dict, Optional

import cv2 as cv
import PyQt5.QtCore as qtc

from gui.workers.worker import Worker
from message.message import (
    IOOperationBody,
    Message,
    Messages,
)
from enums import (
    BODY_KEY,
    FILE_TYPE,
    IO_OPERATION_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)


class FramesExtractionWorker(Worker):

    def __init__(
            self,
            signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
            *args,
            **kwargs
    ):
        super().__init__(signals, *args, **kwargs)

    def process(self, msg: Message):
        data = msg.body.data
        video_path = data[BODY_KEY.VIDEO_PATH]
        data_directory = data[BODY_KEY.DATA_DIRECTORY]
        resize = data[BODY_KEY.RESIZE]
        every_n_th = data[BODY_KEY.EVERY_N_TH_FRAME]

        vidcap = cv.VideoCapture(video_path)
        success, image = vidcap.read()

        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT)) // every_n_th

        msg = Messages.CONFIGURE_WIDGET(
            SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
            WIDGET.JOB_PROGRESS,
            'setMaximum',
            [total_frames],
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

        count = 0
        while success:
            im_path = os.path.join(data_directory, f'frame_{count}.jpg')

            msg = Message(
                MESSAGE_TYPE.REQUEST,
                MESSAGE_STATUS.OK,
                SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
                SIGNAL_OWNER.IO_WORKER,
                IOOperationBody(
                    io_operation_type=IO_OPERATION_TYPE.SAVE,
                    file_path=im_path,
                    file=image,
                    file_type=FILE_TYPE.IMAGE,
                    multipart=True,
                    part=count + 1,
                    total=total_frames,
                )
            )

            if resize:
                msg.body.data[BODY_KEY.RESIZE] = True
                msg.body.data[BODY_KEY.NEW_SIZE] = data[BODY_KEY.NEW_SIZE]

            self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

            for _ in range(every_n_th - 1):
                success, image = vidcap.read()

            success, image = vidcap.read()
            count += 1

            # wait for signal from io worker that it saved picture
            if success:
                _ = self.wait_for_element()
