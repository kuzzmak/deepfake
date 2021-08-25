from __future__ import annotations

import os
from typing import Tuple, Union

import cv2 as cv
import numpy as np

from core.exception import (
    FileDoesNotExistsError,
    NotFileError,
    UnsupportedImageTypeError,
)
from enums import IMAGE_FORMAT


class Image:
    """Simple class for storing image data, format and other things of interest
    when dealing with images.
    """

    def __init__(self, path: str, data: np.ndarray) -> None:
        """Constructor.

        Parameters
        ----------
        path : str
            path to the image
        data : np.ndarray
            image in an array format

        Raises
        ------
        UnsupportedImageTypeError
            trying to load image of the unsupported format
        """
        ext = path.split('.')[-1]
        formats = [f.value for f in IMAGE_FORMAT]
        if ext not in formats:
            raise UnsupportedImageTypeError(ext)

        self._path = path
        self._data = data
        self._format = IMAGE_FORMAT[ext.upper()]
        self._name = os.path.basename(path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def format(self) -> IMAGE_FORMAT:
        return self._format

    @property
    def shape(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        return self.data.shape

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def load(path: str, color: bool = True) -> Image:
        """Loads image from the provided `path`.

        Parameters
        ----------
        path : str
            path to the image
        color : bool, optional
            load image in color or grayscale, by default True

        Returns
        -------
        Image
            image object

        Raises
        ------
        FileDoesNotExistsError
            image on the provided `path` does not exist
        NotFileError
            provided `path` is not a file
        """
        if not os.path.exists(path):
            raise FileDoesNotExistsError(path)

        if not os.path.isfile(path):
            NotFileError(path)

        flags = cv.IMREAD_COLOR
        if not color:
            flags = cv.IMREAD_GRAYSCALE

        data = cv.imread(path, flags)
        image = Image(path, data)

        return image
