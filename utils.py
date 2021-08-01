import os
from typing import List

import numpy as np

import cv2 as cv

import PyQt5.QtGui as qtg


def get_file_paths_from_dir(dir: str) -> List[str] or None:
    """Returns list of paths to files in directory. If no files
    are found in directory, None is returned.

    Parameters
    ----------
    dir : str
        directory with files

    Returns
    -------
    list[str] or None
        list of paths
    """
    if not os.path.exists(dir):
        return None

    files = [f for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    curr_dir = os.path.abspath(dir)
    file_paths = [os.path.join(curr_dir, x) for x in files]

    return file_paths


def qicon_from_path(path: str) -> qtg.QIcon:
    """Loads image from path and makes `qtg.QIcon` object.

    Parameters
    ----------
    path : str
        path to image

    Returns
    -------
    qtg.QIcon
        icon
    """
    return qtg.QIcon(qtg.QPixmap(path))


def np_array_to_qicon(image: np.ndarray) -> qtg.QIcon:
    """Converts `np.ndarray` to `QIcon`.

    Parameters
    ----------
    image : np.ndarray
        image in array format

    Returns
    -------
    qtg.QIcon
        icon
    """
    image = qtg.QImage(
        image.data,
        image.shape[1],
        image.shape[0],
        image.shape[1] * 3,
        qtg.QImage.Format_RGB888)
    image = qtg.QIcon(qtg.QPixmap(image))
    return image


def resize_image_retain_aspect_ratio(image: np.ndarray,
                                     max_img_size_per_dim: int) -> np.ndarray:
    """Resizes image retaining aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        image to resize
    max_img_size_per_dim : int
        maximal height or maximal width of the image

    Returns
    -------
    np.ndarray
        resized image
    """
    height, width, _ = image.shape
    bigger_size = max(height, width)

    scale = max_img_size_per_dim / bigger_size

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    return cv.resize(image, dim, interpolation=cv.INTER_AREA)
