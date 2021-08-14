import os
from typing import Dict, Optional

import PyQt5.QtCore as qtc

import cv2 as cv

from core.face_detection.algorithms.FaceDetectionModel \
    import FaceDetectionModel
from core.face_detection.algorithms.s3fd.a3fd_fdm import S3FDFDM

from enums import (
    BODY_KEY,
    DATA_TYPE,
    FACE_DETECTION_ALGORITHM,
    FILE_TYPE,
    IO_OPERATION_TYPE,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)

from gui.workers.worker import Worker

from message.message import Body, IOOperationBody, Message, Messages

from utils import get_file_paths_from_dir


class FaceDetectionWorker(Worker):

    def __init__(
            self,
            signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
            *args,
            **kwargs
    ):
        super().__init__(signals, *args, **kwargs)

    def process(self, msg: Message):
        data = msg.body.data

        input_data_directory = data[BODY_KEY.INPUT_DATA_DIRECTORY]
        output_data_directory = data[BODY_KEY.OUTPUT_DATA_DIRECTORY]
        input_faces_directory = data[BODY_KEY.INPUT_FACES_DIRECTORY]
        output_faces_directory = data[BODY_KEY.OUTPUT_FACES_DIRECTORY]
        algorithm = data[BODY_KEY.ALGORITHM]
        device = data[BODY_KEY.DEVICE]

        if algorithm == FACE_DETECTION_ALGORITHM.S3FD:
            model = S3FDFDM(device)

        self._process(
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            DATA_TYPE.INPUT,
            input_data_directory,
            input_faces_directory,
            model,
        )

        self._process(
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            DATA_TYPE.OUTPUT,
            output_data_directory,
            output_faces_directory,
            model,
        )

    @staticmethod
    def _process(
        message_worker_sig: qtc.pyqtSignal,
        data_type: DATA_TYPE,
        data_directory: str,
        faces_directory: str,
        model: FaceDetectionModel,
    ):
        if data_directory is not None:
            images = get_file_paths_from_dir(data_directory)

            _msg = Messages.CONFIGURE_WIDGET(
                SIGNAL_OWNER.FACE_DETECTION_WORKER,
                WIDGET.JOB_PROGRESS,
                'setMaximum',
                [len(images)],
            )
            message_worker_sig.emit(_msg)

            images_counter = 0
            img_name = 'if' if data_type == DATA_TYPE.INPUT else 'of'
            img_name += f'_{images_counter}.jpg'
            recipient = SIGNAL_OWNER.DETECTION_ALGORITHM_TAB_INPUT_PICTURE_VIEWER \
                if data_type == DATA_TYPE.INPUT \
                else SIGNAL_OWNER.DETECTION_ALGORITHM_TAB_OUTPUT_PICTURE_VIEWER

            for img_path in images:

                image = cv.imread(img_path, cv.IMREAD_COLOR)
                faces = model.detect(image)

                for face in faces:
                    _msg = Message(
                        MESSAGE_TYPE.REQUEST,
                        MESSAGE_STATUS.OK,
                        SIGNAL_OWNER.FACE_DETECTION_WORKER,
                        recipient,
                        Body(
                            JOB_TYPE.IMAGE_DISPLAY,
                            {
                                BODY_KEY.FILE: face,
                            }
                        )
                    )
                    message_worker_sig.emit(_msg)

                    _msg = Message(
                        MESSAGE_TYPE.REQUEST,
                        MESSAGE_STATUS.OK,
                        SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
                        SIGNAL_OWNER.IO_WORKER,
                        IOOperationBody(
                            io_operation_type=IO_OPERATION_TYPE.SAVE,
                            file_path=os.path.join(
                                faces_directory,
                                img_name,
                            ),
                            file=face,
                            file_type=FILE_TYPE.IMAGE,
                            multipart=True,
                            part=images_counter + 1,
                            total=len(images),
                        )
                    )
                    message_worker_sig.emit(_msg)
