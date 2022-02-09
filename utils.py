import builtins
import errno
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2 as cv
import gdown
import numpy as np
import PyQt5.QtGui as qtg
import torch
from torch.hub import get_dir

from enums import IMAGE_FORMAT, NUMBER_TYPE

logger = logging.getLogger(__name__)


def get_file_extension(file_path: Union[str, Path]) -> str:
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


def get_image_paths_from_dir(dir: Union[str, Path]) -> List[Path] or None:
    """Function for getting all image paths in directory `dir`. Only images
    with extension in `IMAGE_FORMAT` will be returned.

    Parameters
    ----------
    dir : Union[str, Path]
        directory with images

    Returns
    -------
    List[Path] or None
        image paths or None if directory does not exist
    """
    file_paths = get_file_paths_from_dir(dir)
    if file_paths is None:
        return None

    images_exts = set(['.' + f.value for f in IMAGE_FORMAT])
    image_paths = [fp for fp in file_paths
                   if fp.suffix in images_exts]

    return image_paths


def get_file_paths_from_dir(
    dir: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> Optional[List[Path]]:
    """Constructs apsolute file paths of the files in `dir`. If files
    with particular extensions are allowed, then `extensions` argument
    should be also passed as an argument.

    Parameters
    ----------
    dir : Union[str, Path]
        directory with files
    extensions : Optional[List[str]], optional
        files that end with these extension will be included, by default None

    Returns
    -------
    Optional[List[Path]]
        list of file paths is they satisfy `extensions` argument
    """
    if isinstance(dir, str):
        dir = Path(dir)

    if not os.path.exists(dir):
        return None

    files = [
        Path(f) for f in os.listdir(dir) if os.path.isfile(dir.joinpath(f))
    ]
    curr_dir = dir.absolute()
    file_paths = [curr_dir.joinpath(f) for f in files]

    if extensions is None:
        return file_paths

    exts = set(['.' + ext for ext in extensions])

    return [f for f in file_paths if f.suffix in exts]


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
        qtg.QImage.Format_RGB888,
    ).rgbSwapped()
    image = qtg.QIcon(qtg.QPixmap(image))
    return image


def resize_image_retain_aspect_ratio(
    image: np.ndarray,
    max_img_size_per_dim: int,
) -> np.ndarray:
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
        os.makedirs(models_dir)
        logger.debug(f'Creded models folder: {models_dir}.')
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.debug(f'Using existing models folder: {models_dir}.')
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    cached_file = os.path.join(models_dir, filename)
    if not os.path.exists(cached_file):
        logger.info(
            'Model not found locally, downloading from Google drive...')

        url = f'https://drive.google.com/uc?id={model_id}'
        model_dir = os.path.join(models_dir, filename)
        gdown.download(url, model_dir, quiet=True)

        logger.info(
            f'Model downloading finished. Model location: {model_dir}.')

    else:
        logger.debug(f'Using local model instance: {cached_file}.')

    return cached_file


def construct_file_path(path: Union[str, Path]) -> Path:
    """Constructs new file name by adding a number to the end of the
    filename if file on this path already exists.

    Parameters
    ----------
    path : Union[str, Path]
        file path

    Returns
    -------
    str
        file path
    """
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        # ['folder1', 'folder2', 'file.txt']
        parts = list(path.parts)
        # 'file.txt'
        filename_with_ext = parts.pop()
        (filename, ext) = filename_with_ext.split('.')
        # making new file path where numbers are added to the end of the
        # filename if more copies of the same file exist
        counter = 1
        folder = Path(*parts)
        while True:
            new_filename_with_ext = f'{filename}_{str(counter)}.{ext}'
            new_file_path = folder / new_filename_with_ext
            if new_file_path.exists():
                counter += 1
            else:
                return new_file_path
    else:
        return path


def tensor_to_np_image(image: torch.Tensor) -> np.ndarray:
    """Converts image in a form of a `torch.Tensor` into image in `np.ndarray`
    format. In tensor form, image is in 0..1 range so it has to be multiplied
    by 255 in order to be displayed correctly in `Figure`.

    Args:
        image (torch.Tensor): image in tensor form

    Returns:
        np.ndarray: image in numpy form
    """
    img = image.cpu().detach().numpy()
    img = img * 255
    # move number of channels to the last dimension
    img = img.transpose(1, 2, 0)
    img = np.float32(img)
    # image was initially in BGR format, convert to RGB to properly show in
    # maptlotlib canvas
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return np.int32(img)


def parse_number(
    number: str,
    number_type: NUMBER_TYPE = NUMBER_TYPE.INT,
) -> Union[int, float, None]:
    """Function for parsing a number from string.

    Args:
        number (str): number to parse
        number_type (NUMBER_TYPE, optional): parse to int or float. Defaults
            to NUMBER_TYPE.INT.

    Returns:
        Union[int, float, None]: parsed number to int or float or `None` if
            trying to parse something that is not a number
    """
    parse_fun = getattr(builtins, number_type.value)
    try:
        num = parse_fun(number)
    except ValueError:
        return None
    return num


def parse_tuple(
    string: str,
    element_type: NUMBER_TYPE = NUMBER_TYPE.INT,
    separator: str = ',',
) -> Tuple[Any, ...]:
    """Parses some string that into tuple where each element is of type
    `element_type`. Each element that is parsed is separated from another
    element by `separator`.

    Parameters
    ----------
    string : str
        string which is being parsed
    element_type : NUMBER_TYPE, optional
        parse each element to which type, by default NUMBER_TYPE.INT
    separator : str, optional
        with what is each element separated from another, by default ','

    Returns
    -------
    Tuple[Any, ...]
        parsed tuple of values
    """
    split = string.split(separator)
    split = [s.strip() for s in split]
    split = [parse_number(s, element_type) for s in split]
    return tuple(split)
