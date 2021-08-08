import numpy as np


def to_chw_bgr(image: np.ndarray) -> np.ndarray:
    """Transpose image from HWC to CHW and from RBG to BGR.

    Parameters
    ----------
    image : np.ndarray
        an image with HWC and RBG layout

    Returns
    -------
    np.ndarray
        transposed image
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image
