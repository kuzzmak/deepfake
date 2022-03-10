from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Union
from tqdm import tqdm

import cv2 as cv
import numpy as np

from core.dictionary import Dictionary
from serializer.face_serializer import FaceSerializer
from utils import get_file_paths_from_dir


logger = logging.getLogger(__name__)


@dataclass
class AlignerConfiguration:
    """Configuration class for `Aligner`.

    Args:
        metadata_directory (Union[str, Path]): path to the directory with
            `Face` metadata objects
        face_size (int): size of the new image square
    """
    metadata_directory: Union[str, Path]
    face_size: int


class Aligner:
    """Utility class used for aligning image to the mean face or aligning face
    landmarks.

    Args:
        configuration (AlignerConfiguration): configuration class for the
            `Aligner` specifying path to the metadata directory and new size
            of the image when aligned
    """

    def __init__(self, configuration: AlignerConfiguration):
        path = configuration.metadata_directory
        if isinstance(path, str):
            path = Path(path)
        self._metadata_path = path
        self._face_size = configuration.face_size

    @staticmethod
    def align_image(
        image: np.ndarray,
        alignment: np.ndarray,
        image_size: int,
    ) -> np.ndarray:
        """Aligns image based on the `alignment` matrix to the size
        `image_size`.

        Args:
            image (np.ndarray): image to align
            alignment (np.ndarray): alignment matrix
            image_size (int): new image size

        Returns:
            np.ndarray: aligned image
        """
        padding = image_size // 4
        alignment = np.copy(alignment) * image_size
        alignment[:, 2] += padding
        new_size = int(image_size + padding * 2)
        warped = cv.warpAffine(
            image,
            alignment,
            (new_size, new_size),
        )
        return cv.resize(
            warped,
            (image_size, image_size),
            cv.INTER_CUBIC,
        )

    def align_landmarks(self) -> None:
        """Initiates alignment process for the landmarks of `Face` objects in
        directory from the configuration. `landmarks.json` and
        `alignments.json` files must be present in input directory. New file
        with aligned landmarks is generated specific to the image size.
        """
        landmarks_path = self._metadata_path / 'landmarks.json'
        if not os.path.exists(landmarks_path):
            logger.error(
                'landmarks.json file does not exist on location: ' +
                f'{str(self._metadata_path)}.'
            )
            return

        alignments_path = self._metadata_path / 'alignments.json'
        if not os.path.exists(alignments_path):
            logger.error(
                'alignments.json file does not exist on location: ' +
                f'{str(self._metadata_path)}.'
            )
            return

        logger.debug('Loading landmarks.')
        landmarks = Dictionary.load(landmarks_path)
        logger.debug('Landmarks loaded.')
        logger.debug('Loading alignments.')
        alignments = Dictionary.load(alignments_path)
        logger.debug('Alignments loaded.')

        aligned_landmarks = Dictionary()

        metadata_paths = get_file_paths_from_dir(self._metadata_path, ['p'])
        for m_p in tqdm(metadata_paths, desc="Images done"):
            face = FaceSerializer.load(m_p)
            face_size = self._face_size
            padding = face_size // 4
            alignment = np.copy(alignments[face.name]) * face_size
            alignment[:, 2] += padding
            new_size = int(face_size + padding * 2)
            scale = new_size / face_size
            dots = cv.transform(
                landmarks[face.name].reshape(1, -1, 2),
                alignment,
            )
            dots = np.divide(dots.reshape(-1, 2), scale).astype(int)
            aligned_landmarks.add(face.name, dots)

        aligned_landmarks.save(
            self._metadata_path / f'aligned_landmarks_{self._face_size}.json'
        )
