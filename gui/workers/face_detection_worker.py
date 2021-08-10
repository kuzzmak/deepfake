from typing import Dict, List, Optional

import PyQt5.QtCore as qtc

import torch

import numpy as np

import cv2 as cv

from PIL import Image

from enums import (
    BODY_KEY,
    FACE_DETECTION_ALGORITHM,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)

from core.face_detection.algorithms.s3fd.data.config import cfg
from core.face_detection.algorithms.s3fd.s3fd import build_s3fd
from core.face_detection.algorithms.s3fd.utils.augmentations import to_chw_bgr

from gui.workers.worker import Worker

from message.message import Body, Message

from utils import get_file_paths_from_dir

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


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

        input_images = get_file_paths_from_dir(input_data_directory)

        if algorithm == FACE_DETECTION_ALGORITHM.S3FD:
            net = build_s3fd('test', cfg.NUM_CLASSES)
            net.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device('cpu')
                )
            )
            net.eval()

        for img_path in input_images:

            extraced_faces = detect(net, img_path, 0.6)

            for face in extraced_faces:
                msg = Message(
                    MESSAGE_TYPE.REQUEST,
                    MESSAGE_STATUS.OK,
                    SIGNAL_OWNER.FACE_DETECTION_WORKER,
                    SIGNAL_OWNER.DETECTION_ALGORITHM_TAB_INPUT_PICTURE_VIEWER,
                    Body(
                        JOB_TYPE.IMAGE_DISPLAY,
                        {
                            BODY_KEY.FILE: face,
                        }
                    )
                )
                self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)


def detect(net, img_path: str, thresh: float):
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv.INTER_LINEAR)
    # image = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = torch.autograd.Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    img = cv.imread(img_path, cv.IMREAD_COLOR)

    faces = []

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            j += 1
            faces.append((int(pt[0]), int(pt[1]), int(pt[2])), int(pt[3]))

    return extract_faces(faces, img)


def extract_faces(faces: List[tuple], img: np.ndarray):
    extracted_faces = []

    for face in faces:
        x1, y1, x2, y2 = face
        extracted_faces.append(img[y1: y2, x1: x2])

    return extracted_faces
