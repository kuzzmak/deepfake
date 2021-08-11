import os
from typing import Dict, List, Optional

import PyQt5.QtCore as qtc

import torch

import numpy as np

import cv2 as cv

from enums import (
    BODY_KEY,
    DATA_TYPE,
    DEVICE,
    FACE_DETECTION_ALGORITHM,
    FILE_TYPE,
    IO_OPERATION_TYPE,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)

from core.face_detection.algorithms.s3fd.data.config import cfg
from core.face_detection.algorithms.s3fd.s3fd import build_s3fd
from core.face_detection.algorithms.s3fd.utils.augmentations import to_chw_bgr

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
        model_path = data[BODY_KEY.MODEL_PATH]
        device = data[BODY_KEY.DEVICE]

        if device == DEVICE.CUDA:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if algorithm == FACE_DETECTION_ALGORITHM.S3FD:
            net = build_s3fd('test')
            net.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device(device.value)
                )
            )
            net.eval()

        _process(
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            DATA_TYPE.INPUT,
            input_data_directory,
            input_faces_directory,
            net,
            device,
        )

        _process(
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            DATA_TYPE.OUTPUT,
            output_data_directory,
            output_faces_directory,
            net,
            device,
        )


def _process(
    message_worker_sig: qtc.pyqtSignal,
    data_type: DATA_TYPE,
    data_directory: str,
    faces_directory: str,
    net,
    device: DEVICE,
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

            faces = detect(net, img_path, 0.6, device)

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


def detect(net, img_path: str, thresh: float, device: DEVICE):
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(1700 * 1200 / (height * width))
    image = cv.resize(
        img,
        None,
        None,
        fx=max_im_shrink,
        fy=max_im_shrink,
        interpolation=cv.INTER_LINEAR,
    )
    # image = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = torch.autograd.Variable(torch.from_numpy(x).unsqueeze(0))
    if device == DEVICE.CUDA:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    faces = []

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            j += 1
            faces.append((int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])))

    return extract_faces(faces, img)


def extract_faces(faces: List[tuple], img: np.ndarray):
    extracted_faces = []

    for face in faces:
        x1, y1, x2, y2 = face
        extracted_faces.append(img[y1: y2, x1: x2])

    return extracted_faces
