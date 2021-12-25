from typing import Union

import cv2 as cv
import numpy as np

from core.bounding_box import BoundingBox
from core.exception import NoLandmarksError
from core.image.image import Image
from core.landmarks import Landmarks


class Face:
    """Class for everything that has to do with faces. Contains image from
    which faces are extracted, extracted faces, alignments...
    """

    def __init__(self):
        self._raw_image = None
        self._bounding_box = None
        self._detected_face = None
        self._landmarks = None
        self._alignment = None
        self._aligned_landmarks = None
        self._aligned_image = None
        self._aligned_mask = None

    @property
    def raw_image(self) -> Union[Image, None]:
        """Raw image on which was detected face.

        Returns
        -------
        Union[Image, None]
            raw image if it was set, else None
        """
        return self._raw_image

    @property
    def bounding_box(self) -> Union[BoundingBox, None]:
        """Bounding box around the face which is representd by two points.

        Returns
        -------
        Union[BoundingBox, None]
            bounding box if the face detection process was run, None else
        """
        return self._bounding_box

    @property
    def detected_face(self) -> Union[np.ndarray, None]:
        """Detected face by the face detection process.

        Returns
        -------
        Union[np.ndarray, None]
            detected face if the face detection process was run, None else
        """
        return self._detected_face

    @property
    def landmarks(self) -> Union[Landmarks, None]:
        """Detected 68 face points.

        Returns
        -------
        Union[Landmarks, None]
            landmarks if the landmark detection process was run, None else
        """
        return self._landmarks

    @property
    def alignment(self) -> Union[np.ndarray, None]:
        """Alignment matrix which transforms this face into a mean face.

        Returns
        -------
        Union[np.ndarray, None]
            alignment matrix if the alignment process was run, None else
        """
        return self._alignment

    @property
    def aligned_landmarks(self) -> Union[np.ndarray, None]:
        """Landmarks which were aligned to the aligned image based on the
        `alignment`.

        Returns
        -------
        Union[np.ndarray, None]
            aligned landmarks if the alignment process was run, None else
        """
        return self._aligned_landmarks

    @property
    def aligned_image(self) -> Union[np.ndarray, None]:
        """Aligned and resized raw image to the mean face based on the
        `alignment`.

        Returns
        -------
        Union[np.ndarray, None]
            aligned image if the alignment process was run, None else
        """
        return self._aligned_image

    @property
    def aligned_mask(self) -> Union[np.ndarray, None]:
        """Aligned and resized mask to the mean face based on the
        `alignment`. Contains ones on the places where's subjects face.

        Returns
        -------
        Union[np.ndarray, None]
            aligned mask if the alignment process was run, None else
        """
        return self._aligned_mask

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

    @aligned_landmarks.setter
    def aligned_landmarks(self, aligned_landmarks: np.ndarray):
        self._aligned_landmarks = aligned_landmarks

    @aligned_image.setter
    def aligned_image(self, aligned_image: np.ndarray) -> np.ndarray:
        self._aligned_image = aligned_image

    @aligned_mask.setter
    def aligned_mask(self, aligned_mask: np.ndarray) -> np.ndarray:
        self._aligned_mask = aligned_mask

    def masked_face_image(self, aligned: bool = True) -> np.ndarray:
        out = self.aligned_image * self.aligned_mask[..., None]
        return out

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
