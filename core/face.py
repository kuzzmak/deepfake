import cv2 as cv

import numpy as np

from core.bounding_box import BoundingBox
from core.landmarks import Landmarks


class Face:
    """Class for everything that has to do with faces. Contains image from
    which faces are extracted, extracted faces, alignments... Also, all
    operations that should be made on faces, should be implemented here.
    """

    def __init__(
        self,
        raw_image: np.ndarray = None,
        bounding_box: BoundingBox = None,
        detected_face: np.ndarray = None,
        landmarks: Landmarks = None,

    ):
        """Constructor.

        Parameters
        ----------
        raw_image : np.ndarray, optional
            image face, by default None
        bounding_box : BoundingBox, optional
            two dots on the image which are enough to make a bounding box
            containing detected face, by default None
        detected_face : np.ndarray, optional
            detected face in the raw_image, by default None
        landmarks : Landmarks, optional
            object containing face landmark dots, by default None
        """
        self._raw_image = raw_image
        self._bounding_box = bounding_box
        self._detected_face = detected_face
        self._landmarks = landmarks

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
    def landmarks(self) -> Landmarks:
        return self._landmarks

    @raw_image.setter
    def raw_image(self, raw_image: np.ndarray):
        self._raw_image = raw_image

    @bounding_box.setter
    def bounding_box(self, bounding_box: BoundingBox):
        self._bounding_box = bounding_box

    @detected_face.setter
    def detected_face(self, detected_face: np.ndarray):
        self._detected_face = detected_face

    @landmarks.setter
    def landmarks(self, landmarks: Landmarks):
        self._landmarks = landmarks

    def draw_landmarks(self) -> np.ndarray:
        """Draws small dots which represent face landmarks on the raw image
        so it's easier to see if landmarks were detected correctly.

        Returns
        -------
        np.ndarray
            raw image copy with drawn landmarks
        """
        if self.landmarks is None:
            return self.raw_image

        copy = np.copy(self.raw_image)
        for dot in self.landmarks.dots:
            (x, y) = list(map(int, dot))
            copy = cv.circle(copy, (x, y), radius=2,
                             color=(255, 255, 255), thickness=-1)
        return copy
