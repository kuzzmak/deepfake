import cv2 as cv

import numpy as np

from core.bounding_box import BoundingBox
from core.exception import NoLandmarksError
from core.image.image import Image
from core.landmarks import Landmarks


class Face:
    """Class for everything that has to do with faces. Contains image from
    which faces are extracted, extracted faces, alignments... Also, all
    operations that should be made on faces, should be implemented here.
    """

    def __init__(
        self,
        raw_image: Image = None,
        bounding_box: BoundingBox = None,
        detected_face: np.ndarray = None,
        landmarks: Landmarks = None,
        alignment: np.ndarray = None,
    ):
        """Constructor.

        Parameters
        ----------
        raw_image : Image, optional
            image object containing image data and other properties,
            by default None
        bounding_box : BoundingBox, optional
            two dots on the image which are enough to make a bounding box
            containing detected face, by default None
        detected_face : np.ndarray, optional
            detected face in the raw_image, by default None
        landmarks : Landmarks, optional
            object containing face landmark dots, by default None
        alignment : np.ndarray, optional
            alignment which transforms initial image to the mean face
        """
        self._raw_image = raw_image
        self._bounding_box = bounding_box
        self._detected_face = detected_face
        self._landmarks = landmarks
        self._alignment = alignment

    @property
    def raw_image(self) -> Image:
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

    @property
    def alignment(self) -> np.ndarray:
        return self._alignment

    @raw_image.setter
    def raw_image(self, raw_image: Image):
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

    @alignment.setter
    def alignment(self, alignment: np.ndarray):
        self._alignment = alignment

    def draw_landmarks(self) -> np.ndarray:
        """Draws small dots which represent face landmarks on the raw image
        so it's easier to see if landmarks were detected correctly.

        Returns
        -------
        np.ndarray
            raw image copy with drawn landmarks
        """
        if self.landmarks is None:
            return self.raw_image.data

        copy = np.copy(self.raw_image.data)
        for dot in self.landmarks.dots:
            (x, y) = list(map(int, dot))
            copy = cv.circle(copy, (x, y), radius=2,
                             color=(255, 0, 0), thickness=-1)
        return copy

    def _add_landmark_to_mask(
        self,
        mask: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """Adds convex fill to the mask on spaces defined by landmarks. These
        spaces can be specific face landmarks or whole face specified by
        landmarks.

        Parameters
        ----------
        mask : np.ndarray
            mask on which to add convex fill
        landmarks : np.ndarray
            face landmarks

        Returns
        -------
        np.ndarray
            convex fill added on mask
        """
        landmarks = landmarks.astype(np.int32)

        cv.fillConvexPoly(mask, cv.convexHull(landmarks), 1,)
        dilate_h = mask.shape[0] // 64
        dilate_w = mask.shape[1] // 64
        mask = cv.dilate(
            mask,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_h, dilate_w)),
            iterations=1,
        )
        return mask

    @property
    def mask(self) -> np.ndarray:
        """Makes face mask based on the detected landmarks on the whole image.

        Returns
        -------
        np.ndarray
            face mask
        """
        h, w, _ = self.raw_image.shape
        mask = np.zeros((h, w, 1), dtype=np.float32)

        if self.landmarks is None:
            raise NoLandmarksError()

        mask = self._add_landmark_to_mask(mask, self.landmarks.dots)

        return mask
