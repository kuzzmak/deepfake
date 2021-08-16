from enums import CONSOLE_MESSAGE_TYPE
import errno
from message.message import Messages
import os
from typing import List

import numpy as np

import cv2 as cv

import gdown

import PyQt5.QtGui as qtg

from torch.hub import get_dir

from console import Console


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
        image.data.tobytes(),
        image.shape[1],
        image.shape[0],
        image.shape[1] * 3,
        qtg.QImage.Format_RGB888).rgbSwapped()
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

    width = int(width * scale)
    height = int(height * scale)
    dim = (width, height)

    return cv.resize(image, dim, interpolation=cv.INTER_AREA)


def load_file_from_google_drive(model_id: str, filename: str) -> str:
    """Function for getting the model from google drive. If model already
    exists locally in `torch.hub.get_dir()` directory, then this path is
    returned. If model doesn't exist locally, it's downloaded firstly from
    google drive and save on location mentioned before.

    Parameters
    ----------
    model_id : str
        id of the model file on google drive, if
        `https://drive.google.com/file/d/1UPY_Rf8A51Jxxi17_zeN8c0tb4TQgDrU`
        is the link to file of the model, then
        `1UPY_Rf8A51Jxxi17_zeN8c0tb4TQgDrU` is the model_id
    filename : str
        what should be the name of the file, including extension

    Returns
    -------
    str
        paht to the model on disk
    """
    hub_dir = get_dir()
    models_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(models_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    cached_file = os.path.join(models_dir, filename)
    if not os.path.exists(cached_file):
        msg = Messages.CONSOLE_PRINT(
            CONSOLE_MESSAGE_TYPE.LOG, f'{filename} not found locally. ' +
            'Downloading from google drive. Please wait...')
        Console.print(msg)

        url = f'https://drive.google.com/uc?id={model_id}'
        model_dir = os.path.join(models_dir, filename)
        gdown.download(url, model_dir, quiet=True)

        msg = Messages.CONSOLE_PRINT(
            CONSOLE_MESSAGE_TYPE.LOG, 'Done downloading. ' +
            f'Model location: {models_dir}.')
        Console.print(msg)
    else:
        msg = Messages.CONSOLE_PRINT(
            CONSOLE_MESSAGE_TYPE.LOG, f'Using local model instance: {cached_file}.')
        Console.print(msg)
    return cached_file
