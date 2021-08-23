from typing import Tuple


class BoundingBox:
    """Represents coordinates of the detected face on some image. First two
    coordinates represent upper left corner and second two corrdinates
    represent lower right corner of the image.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Constructor.

        Parameters
        ----------
        x1 : int
            x coordinate of the upper left image corner
        y1 : int
            y coordinate of the upper left image corner
        x2 : int
            x coordinate of the lower right image corner
        y2 : int
            y coordinate of the lower right image corner
        """
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    @property
    def upper_left(self) -> Tuple[int, int]:
        return (self._x1, self._y1)

    @property
    def lower_right(self) -> Tuple[int, int]:
        return (self._x2, self._y2)

    @property
    def center(self) -> Tuple[int, int]:
        x = round(self._x2 - (self._x2 - self._x1) / 2.)
        y = round(self._y2 - (self._y2 - self._y1) / 2.)
        return (x, y)

    def __repr__(self):
        return f'{self.upper_left} - {self.lower_right}'
