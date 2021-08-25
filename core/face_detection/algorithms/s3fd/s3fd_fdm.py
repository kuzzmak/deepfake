from typing import List

import cv2 as cv

import numpy as np

import torch

from core.bounding_box import BoundingBox
from core.face import Face
from core.face_detection.algorithms.face_detection_model \
    import FaceDetectionModel
from core.face_detection.algorithms.s3fd.s3fd_model_factory \
    import S3FDModelFactory
from core.face_detection.algorithms.s3fd.utils.augmentations import to_chw_bgr
from core.image.image import Image

from enums import DEVICE


class S3FDFDM(FaceDetectionModel):
    """Face detection model for S3FD algorithm."""

    def __init__(self, device: DEVICE):
        super().__init__(S3FDModelFactory, device)

    def detect_faces(self, image: Image) -> List[Face]:
        thresh = 0.6
        height, width, _ = image.shape
        max_im_shrink = np.sqrt(1700 * 1200 / (height * width))
        img = cv.resize(
            image.data,
            None,
            None,
            fx=max_im_shrink,
            fy=max_im_shrink,
            interpolation=cv.INTER_LINEAR,
        )
        # image = cv2.resize(img, (640, 640))
        x = to_chw_bgr(img)
        x = x.astype('float32')
        x -= np.array([104., 117., 123.])[:, np.newaxis,
                                          np.newaxis].astype('float32')
        x = x[[2, 1, 0], :, :]

        x = torch.from_numpy(x).unsqueeze(0)
        if self.device == DEVICE.CUDA:
            x = x.cuda()

        with torch.no_grad():
            y = self.model(x)
        detections = y.data
        scale = torch.Tensor([width, height, width, height])

        bounding_boxes = []

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                j += 1
                bb = list(map(int, pt))
                bounding_boxes.append(BoundingBox(*bb))

        return self.extract_faces(bounding_boxes, image.data)
