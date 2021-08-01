import os

import cv2 as cv

from gui.workers.worker import Worker

from message.message import (
    ConfigureWidgetMessageBody,
    IO_OperationMessageBody,
    Message,
)

from enums import (
    IO_OP_TYPE,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)


class FramesExtractionWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        video_path, dest_dir, resize, new_size, image_format = msg.body.get_data()

        vidcap = cv.VideoCapture(video_path)
        success, image = vidcap.read()

        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

        msg = Message(
            MESSAGE_TYPE.REQUEST,
            ConfigureWidgetMessageBody(
                WIDGET.JOB_PROGRESS,
                'setMaximum',
                [total_frames]
            )
        )

        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

        count = 0
        while success:
            im_path = os.path.join(dest_dir, f'frame_{count}.jpg')

            msg = Message(
                MESSAGE_TYPE.REQUEST,
                IO_OperationMessageBody(
                    IO_OP_TYPE.SAVE,
                    im_path,
                    None,
                    image,
                    True,
                    count + 1,
                    total_frames,
                )
            )

            self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

            success, image = vidcap.read()
            count += 1
