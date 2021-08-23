import abc

from core.base_model import BaseModel
from core.face import Face, Landmarks
from core.model_factory import ModelFactory

from enums import DEVICE


class LandmarkDetectionModel(BaseModel):
    """Base class which all landmark detectio nalgorithms should implement."""

    def __init__(self, model_factory: ModelFactory, device: DEVICE):
        super().__init__(model_factory, device)

    @abc.abstractmethod
    def detect_landmarks(self, face: Face) -> Landmarks:
        """Detects landmarks on the Face object. Landmarks are dots that
        describe face features like eyes, mouths...

        Parameters
        ----------
        face : Face
            face object with raw image and bounding box

        Returns
        -------
        Landmarks
            object containing landmarks and functionality to easy get
            landmarks from specific face area
        """
        ...
