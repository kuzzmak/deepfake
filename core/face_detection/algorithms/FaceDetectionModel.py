import abc
from core.face_detection.algorithms.ModelFactory import ModelFactory
from typing import List

import numpy as np

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
