import cv2 as cv
import numpy as np

from core.exception import NoLandmarksError
from core.face import Face
from core.face_alignment.utils import get_face_mask, umeyama
from core.landmarks import MEAN_FACE_2D


class FaceAligner:

    @staticmethod
    def align_face(face: Face, image_size: int) -> None:
        """Function for aligning face to the mean face i.e. raw image is
        aligned and resized to the `image_size x image_size`, face landmarks
        are aligned and face mask is also aligned.

        Parameters
        ----------
        face : Face
            face which is getting aligned
        image_size : int
            size of the square image to which face will be aligned
        """
        if face.alignment is None:
            FaceAligner._calculate_alignment(face)
        padding = image_size // 4
        alignment = np.copy(face.alignment) * image_size
        alignment[:, 2] += padding
        new_size = int(image_size + padding * 2)
        FaceAligner._align_face_image(face, alignment, new_size, image_size)
        FaceAligner._align_face_landmarks(
            face,
            alignment,
            new_size,
            image_size,
        )
        FaceAligner._align_mask(face)

    @staticmethod
    def _calculate_alignment(face: Face):
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
    def _align_face_image(
        face: Face,
        alignment,
        new_size: int,
        size: int,
    ) -> None:
        """Alignes raw face image to the new size based on the `alignment` matrix.

        Parameters
        ----------
        face : Face
            face object containing raw image
        alignment : [type]
            alignment matrix
        new_size : int
            size of the image which takes into account image padding
        size : int
            size of the new square image
        """
        warped = cv.warpAffine(
            face.raw_image.data,
            alignment,
            (new_size, new_size),
        )
        face.aligned_image = cv.resize(warped, (size, size), cv.INTER_CUBIC)

    @staticmethod
    def _align_face_landmarks(
        face: Face,
        alignment: np.ndarray,
        new_size: int,
        size: int,
    ) -> None:
        """Function for aligning detected face landmarks.

        Parameters
        ----------
        face : Face
            face object containing landmarks
        alignment : np.ndarray
            alignment matrix
        new_size : int
            size of the image before resizing to size
        size : int
            size of the image after resizing
        """
        scale = new_size / size
        dots = cv.transform(face.landmarks.dots.reshape(1, -1, 2), alignment)
        dots = dots.reshape(-1, 2).astype(int)
        dots = np.divide(dots, scale).astype(int)
        face.aligned_landmarks = dots

    @staticmethod
    def _align_mask(face: Face):
        """Function for aligning face mask.

        Parameters
        ----------
        face : Face
            face object containing aligned landmarks and aligned image
        """
        face.aligned_mask = get_face_mask(
            face.aligned_image,
            face.aligned_landmarks,
        )
