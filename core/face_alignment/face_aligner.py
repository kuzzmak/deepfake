import numpy as np

from core.exception import NoLandmarksError
from core.face import Face
from core.face_alignment.utils import transform, umeyama
from core.landmarks import MEAN_FACE_2D


class FaceAligner:

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
        return FaceAligner._get_aligned(face, image_size, False)

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
        return FaceAligner._get_aligned(face, image_size, True)

    @staticmethod
    def _get_aligned(
        face: Face,
        image_size: int,
        mask: bool = False,
    ) -> np.ndarray:
        """Resizes detected face or face mask to the size `image_size`. If
        face objects contains no alignment, it is being calculated.

        Parameters
        ----------
        face : Face
            face object
        image_size : int
            size of the square to which image is being resized
        mask : bool, optional
            return aligned mask, if false, then aligned face is returned, by
            default False

        Returns
        -------
        np.ndarray
            aligned face or mask
        """
        if face.alignment is None:
            FaceAligner._calculate_alignment(face)

        image = face.raw_image.data

        if mask:
            # Mask contains ones on places where face should be so in order to
            # extract only face from the raw image, mask is multiplied with
            # the raw image so only pixels on places where mask is one, appear
            # in final image. OpenCV uses H,W,C convention so image is
            # transposed in order to be able to do matrix multiplication.
            image = image.transpose(2, 0, 1) * face.mask
            image = image.transpose(1, 2, 0).astype('uint8')

        padding = image_size // 4
        aligned_image = transform(image, face.alignment, image_size, padding)
        return aligned_image
