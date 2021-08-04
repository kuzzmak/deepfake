import os

import cv2 as cv

from gui.workers.worker import Worker

from message.message import (
    # ConfigureWidgetMessageBody,
    # IO_OperationMessageBody,
    Body,
    IOOperationBody,
    Message,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):

        data = msg.body.data
        video_path = data[BODY_KEY.VIDEO_PATH]
        data_directory = data[BODY_KEY.DATA_DIRECTORY]
        resize = data[BODY_KEY.RESIZE]

        vidcap = cv.VideoCapture(video_path)
        success, image = vidcap.read()

        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

        # msg = Message(
        #     MESSAGE_TYPE.REQUEST,
        #     ConfigureWidgetMessageBody(
        #         WIDGET.JOB_PROGRESS,
        #         'setMaximum',
        #         [total_frames]
        #     )
        # )

        # self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

        count = 0
        while success:
            im_path = os.path.join(data_directory, f'frame_{count}.jpg')

            msg = Message(
                MESSAGE_TYPE.REQUEST,
                MESSAGE_STATUS.OK,
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

            success, image = vidcap.read()
            count += 1
