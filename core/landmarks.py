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
