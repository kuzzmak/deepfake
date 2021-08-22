import abc
from typing import List

import numpy as np

from core.bounding_box import BoundingBox
from core.face_detection.algorithms.ModelFactory import ModelFactory

from enums import DEVICE


class FaceDetectionModelMeta(abc.ABCMeta):
    ...


class FaceDetectionModel(metaclass=FaceDetectionModelMeta):
    """Base class which every face detection algorithm should implement.
    """

    def __init__(self, model_factory: ModelFactory, device: DEVICE):
        """Constructor.

        Parameters
        ----------
        model_factory : ModelFactory
            class of the model factory
        device : DEVICE
            computation device
        """
        self.device = device
        self.model = model_factory.build_model(device)
        self.model.eval()
        self.model.to(device.value)

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """If any face exists in image, this method should detect
        all of them.

        Parameters
        ----------
        image : np.ndarray
            image for detection

        Returns
        -------
        List[np.ndarray]
            list of detected faces
        """
        ...

    @staticmethod
    def extract_faces(bounding_boxes: List[BoundingBox], img: np.ndarray) -> List[np.ndarray]:
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
        List[np.ndarray]
            list of faces
        """
        extracted_faces = []

        for bb in bounding_boxes:
            (x1, y1), (x2, y2) = bb.upper_left, bb.lower_right
            extracted_faces.append(img[y1:y2, x1:x2])

        return extracted_faces
