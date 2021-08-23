from core.bounding_box import BoundingBox
import numpy as np


class Landmarks:
    """Class containing 68 landmarks detected by some landmark detection
    algorithm. Provides easy access to particular landmark areas on face
    like eyes, mouth, nose...
    """

    _face_dots = slice(0, 17)
    _eyebrow1_dots = slice(17, 22)
    _eyebrow2_dots = slice(22, 27)
    _nose_dots = slice(27, 31)
    _nostril_dots = slice(31, 36)
    _eye1_dots = slice(36, 42)
    _eye2_dots = slice(42, 48)
    _lips_dots = slice(48, 60)
    _teeth_dots = slice(60, 68)

    def __init__(self, landmarks: np.ndarray) -> None:
        """Constructor.

        Parameters
        ----------
        landmarks : np.ndarray
            68 landmarks detected by landmark detection algorithm
        """
        self._dots = landmarks

    @property
    def dots(self) -> np.ndarray:
        return self._dots

    @property
    def face(self) -> np.ndarray:
        return self.dots[self._face_dots]

    @property
    def eyebrow1(self) -> np.ndarray:
        return self.dots[self._eye1_dots]

    @property
    def eyebrow2(self) -> np.ndarray:
        return self.dots[self._eye2_dots]

    @property
    def nose(self) -> np.ndarray:
        return self.dots[self._nose_dots]

    @property
    def nostril(self) -> np.ndarray:
        return self.dots[self._nostril_dots]

    @property
    def eye1(self) -> np.ndarray:
        return self.dots[self._eye1_dots]

    @property
    def eye2(self) -> np.ndarray:
        return self.dots[self._eye2_dots]

    @property
    def lips(self) -> np.ndarray:
        return self.dots[self._lips_dots]

    @property
    def teeth(self) -> np.ndarray:
        return self.dots[self._teeth_dots]


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
            array of dots representing face, by default None
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
    def landmarks(self) -> np.ndarray:
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
    def landmarks(self, landmarks: np.ndarray):
        self._landmarks = landmarks
