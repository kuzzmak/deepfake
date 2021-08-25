import numpy as np

from core.exception import NoLandmarksError
from core.face import Face
from core.face_alignment.utils import transform, umeyama
from core.landmarks import MEAN_FACE_2D


class FaceAligner:

    @staticmethod
    def calculate_alignment(face: Face):
        """Calculates alignment matrix which transforms face from the raw
        image (detected face) into the mean face. Alignment matrix is set
        to the `alignment` atribute of the `face` input object.

        Parameters
        ----------
        face : Face
            face object containing raw image and detected face landmarks
        """
        if face.landmarks is None:
            raise NoLandmarksError()

        landmarks = face.landmarks.no_face
        alignment = umeyama(landmarks, MEAN_FACE_2D, True)[0:2]
        face.alignment = alignment

    @staticmethod
    def get_aligned_face(face: Face, image_size: int) -> np.ndarray:
        """Transforms face from the raw image into the cropped image where
        face is centered. If `face` object has no alignments, they are being
        calculated and then transformation is done.

        Parameters
        ----------
        face : Face
            face object containing necesary data for face alignment
        image_size : int
            size of the square in which face will be transformed

        Returns
        -------
        np.ndarray
            aligned image of the face
        """
        if face.alignment is None:
            FaceAligner.calculate_alignment(face)

        padding = image_size // 4
        aligned_image = transform(
            face.raw_image.data,
            face.alignment,
            image_size,
            padding,
        )
        return aligned_image

    @staticmethod
    def get_aligned_mask(face: Face, image_size: int) -> np.ndarray:
        """Transforms mask which is taken from the `face` object into a
        centered one and cropped to the `image_size`.

        Parameters
        ----------
        face : Face
            face object containing necessary data
        image_size : int
            size of the square in which face mask will be transformed

        Returns
        -------
        np.ndarray
            transformed mask
        """
        if face.alignment is None:
            FaceAligner.calculate_alignment(face)

        mask = face.mask
        padding = image_size // 4
        aligned_mask = transform(mask, face.alignment, image_size, padding)
        return aligned_mask
