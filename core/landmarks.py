import numpy as np

_mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483,
    0.799124, 0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127,
    0.36688, 0.426036, 0.490127, 0.554217, 0.613373, 0.121737, 0.187122,
    0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428, 0.793132,
    0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
    0.551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
    0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364,
    0.490127, 0.42689])

_mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625,
    0.587326, 0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758,
    0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852, 0.178758,
    0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
    0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
    0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182,
    0.831803, 0.824182])

MEAN_LANDMARKS_2D = np.stack([_mean_face_x, _mean_face_y], axis=1)


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
