from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import cv2 as cv
import numpy as np

from core.dictionary import Dictionary
from serializer.face_serializer import FaceSerializer
from utils import get_file_paths_from_dir


logger = logging.getLogger(__name__)


@dataclass
class AlignerConfiguration:
    faces_directory: str
    face_size: int


def _map_to_numpy_array(data: Dict[str, list]) -> Dict[str, np.ndarray]:
    return dict(zip(data, map(lambda v: np.array(v), data.values())))


def _load(path: Path) -> Dict[str, np.ndarray]:
    with open(path, 'r') as f:
        data = json.load(f)
    return _map_to_numpy_array(data)


class Aligner:

    def __init__(self, configuration: AlignerConfiguration):
        self._conf = configuration

    def align(self) -> None:
        metadata_path = Path(self._conf.faces_directory)
        parent = metadata_path.parent
        training_data_directory = parent / 'training_data'
        if not os.path.exists(training_data_directory):
            os.makedirs(training_data_directory)

        landmarks_path = metadata_path / 'landmarks.json'
        if not os.path.exists(landmarks_path):
            logger.error(
                'landmarks.json file does not exist on location: ' +
                f'{str(metadata_path)}.'
            )
            return

        alignments_path = metadata_path / 'alignments.json'
        if not os.path.exists(alignments_path):
            logger.error(
                'alignments.json file does not exist on location: ' +
                f'{str(metadata_path)}.'
            )
            return

        logger.debug('Loading landmarks.')
        landmarks = _load(landmarks_path)
        logger.debug('Landmarks loaded.')
        logger.debug('Landmarks alignments.')
        alignments = _load(alignments_path)
        logger.debug('Alignments loaded.')

        metadata_paths = get_file_paths_from_dir(metadata_path, ['p'])

        aligned_landmarks = Dictionary()

        for m_p in tqdm(metadata_paths, desc="Images done"):
            face = FaceSerializer.load(m_p)
            face_size = self._conf.face_size
            padding = face_size // 4
            alignment = np.copy(alignments[face.name]) * face_size
            alignment[:, 2] += padding
            new_size = int(face_size + padding * 2)

            warped = cv.warpAffine(
                face.raw_image.data,
                alignment,
                (new_size, new_size),
            )
            aligned_image = cv.resize(
                warped,
                (face_size, face_size),
                cv.INTER_CUBIC,
            )

            im_name = face.name.split('.')[0] + '.jpg'
            im_path = training_data_directory / f'{im_name}'
            cv.imwrite(str(im_path), aligned_image)

            # cv.imshow('im', face.detected_face)
            # cv.waitKey()

            scale = new_size / face_size
            dots = cv.transform(
                landmarks[face.name].reshape(1, -1, 2),
                alignment,
            )
            dots = np.divide(dots.reshape(-1, 2), scale).astype(int)
            aligned_landmarks.add(im_name, dots)

        aligned_landmarks.save(training_data_directory / 'landmarks.json')
