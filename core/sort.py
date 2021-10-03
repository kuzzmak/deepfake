from typing import Callable, List, Tuple

import cv2 as cv
import imagehash
from PIL import Image

from core.face import Face


def _calculate_hashes(
    faces: List[Face],
    hash_functions: List[Callable],
) -> List[Tuple]:
    """Calculates image hashes which are passed as an argument for all faces.

    Args:
        faces (List[Face]): list of `Face` metadata objects for which image
            hashes are calculated
        hash_functions (List[Callable]): list of image hash functions

    Returns:
        List[Tuple]: list of tuples for every face where tuple contains hash
            for every image hash function
    """
    hashes = []
    for face in faces:
        rgb = cv.cvtColor(face.detected_face, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        curr_hashes = tuple(hf(image) for hf in hash_functions)
        hashes.append(curr_hashes)
    return hashes


def sort_faces_by_image_hash(
    faces: List[Face],
    eps: int,
) -> Tuple[List[int], List[int]]:
    """Sorts face metadata objects from the list `faces` Metadata is
    read from the directory, image hashes are calculated and only ones that
    don't differ to much from the previous one are marked as satisfactory.

    Args:
        faces (List[Face]): list of `Face` metadata objects
        eps (int): constant which limits dissimilarity between two hashes

    Returns:
        Tuple[List[int], List[int]]: list of indices that satisfy, list of
            indices that do not satisfy
    """
    hashes = _calculate_hashes(
        faces,
        [imagehash.dhash, imagehash.dhash_vertical],
    )

    old_v, old_h = hashes[0]
    # images satisfying image hash similarity and difference between current
    # and previous one is less than eps
    indices_ok = []
    indices_not_ok = []
    for i, hash in enumerate(hashes):
        hash_v, hash_h = hash
        # difference between old hash and current one
        diff_v = abs(old_v - hash_v)
        diff_h = abs(old_h - hash_h)

        if diff_v <= eps or diff_h <= eps:
            indices_ok.append(i)
            # replacing old value with new one because of the flow of the
            # video, reasoning here is that frames change little from one
            # frame to another
            old_v = hash_v
            old_h = hash_h
        else:
            indices_not_ok.append(i)

    return indices_ok, indices_not_ok
