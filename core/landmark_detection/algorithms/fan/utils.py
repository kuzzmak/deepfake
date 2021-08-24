from typing import Tuple, Union

import cv2 as cv
from numba import jit
import numpy as np
import torch


def transform(
        point: torch.Tensor,
        center: Union[torch.Tensor, np.ndarray],
        scale: float,
        resolution: float,
        invert: bool = False,
) -> torch.Tensor:
    """Generate an affine transformation matrix. Given a set of points, a
    center, a scale and a targer resolution, the function generates and
    affine transformation matrix. If invert is ``True`` it will produce
    the inverse transformation.

    Parameters
    ----------
    point : torch.Tensor
        the input 2D point
    center : Union[torch.Tensor, np.ndarray]
        the center around which to perform the transformations
    scale : float
        the scale of the face/object
    resolution : float
        the output resolution
    invert : bool, optional
        should transformation be inverted, by default False

    Returns
    -------
    torch.Tensor
        transformation matrix
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop(
    image: np.ndarray,
    center: np.ndarray,
    scale: float,
    resolution: float = 256.0,
) -> np.ndarray:
    """Center crops an image or set of heatmaps.

    Parameters
    ----------
    image : np.ndarray
        rgb image
    center : np.ndarray
        the center of the object, usually the same as of the bounding box
    scale : float
        scale of the face
    resolution : float, optional
        the size of the output cropped image, by default 256.0

    Returns
    -------
    np.ndarray
        cropped input image
    """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)

    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv.resize(newImg, dsize=(int(resolution), int(resolution)),
                       interpolation=cv.INTER_LINEAR)
    return newImg


@jit(nopython=True)
def transform_np(
    point: np.ndarray,
    center: np.ndarray,
    scale: float,
    resolution: float,
    invert: bool = False,
) -> np.ndarray:
    """Generate an affine transformation matrix. Given a set of points, a
    center, a scale and a targer resolution, the function generates and
    affine transformation matrix. If invert is ``True`` it will produce
    the inverse transformation.

    Parameters
    ----------
    point : np.ndarray
        the input 2D point
    center : np.ndarray
        the center around which to perform the transformations
    scale : float
        the scale of the face/object
    resolution : float
        the output resolution
    invert : bool, optional
        should transformation be inverted, by default False

    Returns
    -------
    np.ndarray
        transformation matrix
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.ascontiguousarray(np.linalg.pinv(t))

    new_point = np.dot(t, _pt)[0:2]

    return new_point.astype(np.int32)


def get_preds_fromhm(
    hm: np.ndarray,
    center: torch.Tensor = None,
    scale: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Parameters
    ----------
    hm : np.ndarray
        the predicted heatmaps, of shape [B, N, W, H]
    center : torch.Tensor, optional
        the center of the bounding box, by default None
    scale : float, optional
        face scale, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        landmarks from heatmap
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = np.argmax(hm_reshape, axis=-1)
    scores = np.take_along_axis(hm_reshape, np.expand_dims(
        idx, axis=-1), axis=-1).squeeze(-1)
    preds, preds_orig = _get_preds_fromhm(hm, idx, center, scale)

    return preds, preds_orig, scores


@jit(nopython=True)
def _get_preds_fromhm(
    hm: np.ndarray,
    idx: torch.Tensor,
    center: torch.Tensor = None,
    scale: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center and the scale is
    provided the function will return the points also in the original
    coordinate frame.

    Parameters
    ----------
    hm : np.ndarray
        the predicted heatmaps, of shape [B, N, W, H]
    idx : np.ndarray
        index
    center : torch.Tensor, optional
        the center of the bounding box, by default None
    scale : float, optional
        face scale, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        landmarks from heatmap
    """
    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += np.sign(diff) * 0.25

    preds -= 0.5

    preds_orig = np.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(
                    preds[i, j], center, scale, H, True)

    return preds, preds_orig
