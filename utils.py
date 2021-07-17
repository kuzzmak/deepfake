import os
from typing import List

import numpy as np

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
