import numpy as np

from core.face import Face
from core.face_alignment.utils import transform, umeyama
from core.landmarks import MEAN_LANDMARKS_2D


class FaceAligner:

    @staticmethod
    def align(face: Face, image_size: int) -> np.ndarray:
        """Transforms raw image with the detected face into cropped image
        where face is aligned to the center of the image. Eyes are aligned
        horizontally and nose is aligned vertically. This alignment is based
        on the detected face landmarks.

        Parameters
        ----------
        face : Face
            face object containing raw image and detected face landmarks
        image_size : int
            new size of the image after transformations

        Returns
        -------
        np.ndarray
            transformed and aligned image
        """
        landmarks = face.landmarks.no_face
        alignment = umeyama(
            landmarks, MEAN_LANDMARKS_2D, True)[0:2]
        padding = image_size // 4
        aligned_image = transform(
            face.raw_image, alignment, image_size, padding)
        return aligned_image
