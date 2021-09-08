import logging
import errno
import os
from typing import List, Optional

import cv2 as cv
import gdown
import numpy as np
import PyQt5.QtGui as qtg
from torch.hub import get_dir

from enums import IMAGE_FORMAT

logger = logging.getLogger(__name__)


def get_file_extension(file_path: str) -> str:
    """Gets file extension.

    Parameters
    ----------
    file_path : str
        path of the file

    Returns
    -------
    str
        file extension
    """
    ext = file_path.split('.')[-1]
    return ext


def get_image_paths_from_dir(dir: str) -> List[str] or None:
    """Function for getting all image paths in directory `dir`. Only images
    with extension in `IMAGE_FORMAT` will be returned.

    Parameters
    ----------
    dir : str
        directory with images

    Returns
    -------
    List[str] or None
        image paths or None if directory does not exist
    """
    file_paths = get_file_paths_from_dir(dir)
    if file_paths is None:
        return None

    images_exts = [f.value for f in IMAGE_FORMAT]
    image_paths = [fp for fp in file_paths
                   if get_file_extension(fp) in images_exts]

    return image_paths


def get_file_paths_from_dir(
    dir: str,
    extensions: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Constructs file apsolute file paths of the files in `dir`. If files
    with particular extensions are allowed, then `extensions` argument
    should be also passed to function.

    Parameters
    ----------
    dir : str
        directory with files
    extensions : Optional[List[str]], optional
        files that end with these extension will be included, by default None

    Returns
    -------
    Optional[List[str]]
        list of file paths is they satisfy `extensions` argument
    """
    if not os.path.exists(dir):
        return None

    files = [f for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    curr_dir = os.path.abspath(dir)
    file_paths = [os.path.join(curr_dir, x) for x in files]

    if extensions is None:
        return file_paths

    exts = set(extensions)
    file_paths = [f for f in file_paths if get_file_extension(f) in exts]

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


# TODO replace gdown with implementation that updates work progress in gui
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
        logger.debug('No models folder, creating...')
        os.makedirs(models_dir)
        logger.debug(f'New models folder: {models_dir}')
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    cached_file = os.path.join(models_dir, filename)
    if not os.path.exists(cached_file):
        logger.debug(
            'Model not found locally, downloading from Google drive...')

        url = f'https://drive.google.com/uc?id={model_id}'
        model_dir = os.path.join(models_dir, filename)
        gdown.download(url, model_dir, quiet=True)

        logger.debug(
            f'Model downloading finished. Model location: {model_dir}.')

    else:
        logger.debug(f'Using local model instance: {cached_file}.')

    return cached_file


def construct_file_path(file_path: str) -> str:
    """Constructs new file name by adding a number to the end of the
    filename if file on this path already exists.

    Parameters
    ----------
    file_path : str
        file path

    Returns
    -------
    str
        file path
    """
    if os.path.exists(file_path):
        # ['folder1', 'folder2', 'file.txt']
        parts = file_path.split(os.sep)
        # 'file.txt'
        filename_with_ext = parts.pop()

        (filename, ext) = filename_with_ext.split('.')

        # making new file path where numbers are added to the end of the
        # filename if more copies of the same file exist
        counter = 1
        while True:
            new_filename_with_ext = f'{filename}_{str(counter)}.{ext}'
            folders = f'{os.sep}'.join(parts)
            new_file_path = os.path.join(folders, new_filename_with_ext)

            if os.path.exists(new_file_path):
                counter += 1
            else:
                return new_file_path
    else:
        return file_path
