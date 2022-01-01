from typing import Union

import cv2 as cv
import numpy as np

from core.bounding_box import BoundingBox
from core.image.image import Image
from core.landmarks import Landmarks


class Face:
    """Class for everything that has to do with faces. Contains image from
    which faces are extracted, extracted faces, alignments...
    """

    def __init__(self):
        self._raw_image = None
        self._mask = None
        self._bounding_box = None
        self._detected_face = None
        self._landmarks = None
        self._alignment = None
        self._aligned_landmarks = None
        self._aligned_image = None
        self._aligned_mask = None
        self._path = None

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
    def mask(self) -> Union[np.ndarray, None]:
        """Binary mask which multiplied with the raw image only gives image
        where only face is visible without background i.e. convex polygon
        defined by landmarks is only visible on raw image.

        Returns
        -------
        Union[np.ndarray, None]
            binary mask for raw image
        """
        return self._mask

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

    @property
    def path(self) -> Union[str, None]:
        """Path for this particular `Face` object if it was saved to the disk.

        Returns:
            Union[str, None]: path if it was saved, None else
        """
        return self._path

    @raw_image.setter
    def raw_image(self, raw_image: Image) -> None:
        self._raw_image = raw_image

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self._mask = mask

    @bounding_box.setter
    def bounding_box(self, bounding_box: BoundingBox) -> None:
        self._bounding_box = bounding_box

    @detected_face.setter
    def detected_face(self, detected_face: np.ndarray) -> None:
        self._detected_face = detected_face

    @landmarks.setter
    def landmarks(self, landmarks: Landmarks) -> None:
        self._landmarks = landmarks

    @alignment.setter
    def alignment(self, alignment: np.ndarray) -> None:
        self._alignment = alignment

    @aligned_landmarks.setter
    def aligned_landmarks(self, aligned_landmarks: np.ndarray) -> None:
        self._aligned_landmarks = aligned_landmarks

    @aligned_image.setter
    def aligned_image(self, aligned_image: np.ndarray) -> None:
        self._aligned_image = aligned_image

    @aligned_mask.setter
    def aligned_mask(self, aligned_mask: np.ndarray) -> None:
        self._aligned_mask = aligned_mask

    @path.setter
    def path(self, path: str) -> None:
        self._path = path

    def masked_face_image(self, aligned: bool = True) -> np.ndarray:
        """Multiplies aligned image with the aligned mask and the result is
        the visible face defined by face landmarks.

        Parameters
        ----------
        aligned : bool, optional
            should this be applied on aligned image and mask or on raw image
                and mask, by default True

        Returns
        -------
        np.ndarray
            image where only face is visible, without background
        """
        face = self.aligned_image
        face_mask = self.aligned_mask
        if not aligned:
            face = self.raw_image.data
            face_mask = self.mask
        masked = np.zeros_like(face, dtype=np.uint8)
        for i in range(3):
            masked[:, :, i] = np.multiply(face[:, :, i], face_mask)
        return masked

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
