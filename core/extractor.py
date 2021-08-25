from typing import List

import numpy as np

from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from core.image.image import Image
from core.landmark_detection.landmark_detection_model \
    import LandmarkDetectionModel
from core.face_detection.algorithms.face_detection_model \
    import FaceDetectionModel


class Extractor:

    def __init__(
        self,
        fdm: FaceDetectionModel,
        ldm: LandmarkDetectionModel,
    ) -> None:
        self.fdm = fdm
        self.ldm = ldm

    def detect_faces(self, image: Image) -> List[Face]:
        """Initiates face detection process on the `image`. When face is
        detected, `Face` object is created and `bounding_box` property is set
        which bounds a face with two dots, upper left and lower right and they
        make a bounding rectangle.

        Parameters
        ----------
        image : Image
            image object with potential faces

        Returns
        -------
        List[Face]
            list of detected `Face` objects
        """
        faces = self.fdm.detect_faces(image)
        for f in faces:
            f.raw_image = image
        return faces

    def detect_landmarks(self, face: Face) -> None:
        """Initiates process of face landmark detection on the `face` object.
        After detection is done `landmarks` property is set on the `face`
        object.

        Note: face detection prior to this process has to be done in order to
        obtain bounding boxes of the potential faces.

        Parameters
        ----------
        face : Face
            face object containing bounding boxes
        """
        landmarks = self.ldm.detect_landmarks(face)
        face.landmarks = landmarks

    def get_aligned_face(self, face: Face, image_size: int) -> np.ndarray:
        """Aligned and centered face getter.

        Parameters
        ----------
        face : Face
            face object containing everything necessary
        image_size : int
            size of the square in which image will be resized

        Returns
        -------
        np.ndarray
            aligned face
        """
        aligned_face = FaceAligner.get_aligned_face(face, image_size)
        return aligned_face

    def get_aligned_mask(self, face: Face, image_size: int) -> np.ndarray:
        """Aligned and centered face mask getter.

        Parameters
        ----------
        face : Face
            face object containing everything necessary
        image_size : int
            size of the square in which image will be resized

        Returns
        -------
        np.ndarray
            aligned face mask
        """
        aligned_mask = FaceAligner.get_aligned_mask(face, image_size)
        return aligned_mask
