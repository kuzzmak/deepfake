from typing import List

import numpy as np

import torch

from core.face_detection.algorithms.FaceDetectionModel \
    import FaceDetectionModel
from core.face_detection.algorithms.faceboxes.faceboxes_model_factory \
    import FaceboxesModelFactory
from core.face_detection.algorithms.faceboxes.layers.functions.prior_box \
    import PriorBox
from core.face_detection.algorithms.utils.bbox_utils import decode
from core.face_detection.algorithms.faceboxes.utils.nms_wrapper import nms

from enums import DEVICE


class FaceboxesFDM(FaceDetectionModel):
    """Face detection model for faceboxes algorithm.
    """

    def __init__(self, device: DEVICE):
        super().__init__(FaceboxesModelFactory, device)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        img = np.float32(image)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device.value)
        scale = scale.to(self.device.value)

        with torch.no_grad():
            loc, conf = self.model(img)  # forward pass
        priorbox = PriorBox((im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device.value)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > 0.05)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, 0.3, self.device.value)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:750, :]

        faces = []

        # show image
        for b in dets:
            if b[4] < 0.5:
                continue
            b = list(map(int, b))
            b = b[:4]
            faces.append(b)

        return self.extract_faces(faces, image)
