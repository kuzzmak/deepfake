from enums import IO_OP_TYPE, MESSAGE_TYPE, SIGNAL_OWNER
import os

import cv2 as cv

from message.message import IO_OperationMessageBody, Message

from gui.workers.worker import Worker


class FramesExtractionWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):
        video_path, dest_dir, image_format = msg.body.get_data()

        vidcap = cv.VideoCapture(video_path)
        success, image = vidcap.read()

        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

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
                )
            )

            self.signals[SIGNAL_OWNER.IO_WORKER].emit(msg)

            success, image = vidcap.read()
            count += 1
