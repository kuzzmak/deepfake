from core.exception import NoBoundingBoxError
import numpy as np

import torch

from core.face import Face
from core.landmarks import Landmarks
from core.landmark_detection.algorithms.fan.fan_model_factory \
    import FANModelFactory
from core.landmark_detection.landmark_detection_model \
    import LandmarkDetectionModel
from core.landmark_detection.algorithms.fan.utils \
    import (
        crop,
        get_preds_fromhm,
    )

from enums import DEVICE


class FANLDM(LandmarkDetectionModel):

    def __init__(self, device: DEVICE) -> None:
        super().__init__(FANModelFactory, device)

    def detect_landmarks(self, face: Face) -> Landmarks:
        if face.bounding_box is None:
            raise NoBoundingBoxError()

        landmarks = self._get_landmarks_from_image(face)
        landmarks = Landmarks(landmarks)
        return landmarks

    def _get_landmarks_from_image(self, face: Face) -> np.ndarray:
        """Predicts 68 different landmarks on the image with face.

        Parameters
        ----------
        face : Face
            Face object containing raw image and bounding box

        Returns
        -------
        np.ndarray
            array of landmarks
        """
        bb = face.bounding_box
        d = (*bb.upper_left, *bb.lower_right)
        center = torch.tensor([*bb.center])
        center[1] = center[1] - (d[3] - d[1]) * 0.12
        scale = (d[2] - d[0] + d[3] - d[1]) / 195.

        inp = crop(face.raw_image, center, scale)
        inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()

        inp = inp.to(self.device.value)
        inp.div_(255.0).unsqueeze_(0)

        with torch.no_grad():
            out = self.model(inp)
        out = out.cpu().numpy()

        _, pts_img, _ = get_preds_fromhm(out, center.numpy(), scale)
        pts_img = torch.from_numpy(pts_img)
        pts_img = pts_img.view(68, 2)

        return pts_img.numpy()
