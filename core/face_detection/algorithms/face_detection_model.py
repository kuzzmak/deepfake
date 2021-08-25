import abc
from typing import List

import numpy as np

from core.base_model import BaseModel
from core.bounding_box import BoundingBox
from core.face import Face
from core.image.image import Image
from core.model_factory import ModelFactory

from enums import DEVICE


class FaceDetectionModel(BaseModel):
    """Base class which every face detection algorithm should implement."""

    def __init__(self, model_factory: ModelFactory, device: DEVICE):
        super().__init__(model_factory, device)

    @abc.abstractmethod
    def detect_faces(self, image: Image) -> List[Face]:
        """If any face exists in image, this method should detect
        all of them.

        Parameters
        ----------
        image : Image
            image for detection

        Returns
        -------
        List[Face]
            list of detected faces
        """
        ...

    @staticmethod
    def extract_faces(
        bounding_boxes: List[BoundingBox],
        img: np.ndarray,
    ) -> List[Face]:
        """Helper function for getting faces out of the image when
        bounding boxes are found by face detection algorithms.

        Parameters
        ----------
        bounding_boxes : List[BoundingBox]
            list of bounding boxes, first two numbers in tuple are
            upper left image corner and second two are lower right corner
        img : np.ndarray
            image from which faces should be extracted

        Returns
        -------
        List[Face]
            list of faces
        """
        extracted_faces = []

        for bb in bounding_boxes:
            (x1, y1), (x2, y2) = bb.upper_left, bb.lower_right

            f = Face()
            f.bounding_box = bb
            f.detected_face = img[y1:y2, x1:x2]

            extracted_faces.append(f)

        return extracted_faces
