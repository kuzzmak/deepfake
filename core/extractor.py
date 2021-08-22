import numpy as np
from typing import List

from core.face import Face
from core.face_detection.algorithms.FaceDetectionModel \
    import FaceDetectionModel


class Extractor:

    def __init__(
        self,
        fdm: FaceDetectionModel,
        # ldm: LandmarkDetectionModel,
    ) -> None:
        self.fdm = fdm

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        faces = self.fdm.detect(image)
        return faces

    def detect_landmarks(self):
        ...
