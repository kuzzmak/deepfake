from core.bounding_box import BoundingBox
import numpy as np


class Face:

    def __init__(
        self,
        raw_image: np.ndarray = None,
        bounding_box: BoundingBox = None,
        detected_face: np.ndarray = None,
        alignments: np.ndarray = None,

    ):
        self._raw_image = raw_image
        self._bounding_box = bounding_box
        self._detected_face = detected_face
        self._alignments = alignments

    @property
    def raw_image(self) -> np.ndarray:
        return self._raw_image

    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box

    @property
    def detected_face(self) -> np.ndarray:
        return self._detected_face

    @property
    def alignments(self) -> np.ndarray:
        return self._alignments

    @raw_image.setter
    def raw_image(self, raw_image: np.ndarray):
        self._raw_image = raw_image

    @bounding_box.setter
    def bounding_box(self, bounding_box: BoundingBox):
        self._bounding_box = bounding_box

    @detected_face.setter
    def detected_face(self, detected_face: np.ndarray):
        self._detected_face = detected_face

    @alignments.setter
    def alignments(self, alignments: np.ndarray):
        self._alignments = alignments
