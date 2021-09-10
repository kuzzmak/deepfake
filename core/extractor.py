import argparse
import logging
import os
from typing import List, Union

import numpy as np
from tqdm import tqdm

from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from core.face_detection.algorithms.faceboxes.faceboxes_fdm import FaceboxesFDM
from core.face_detection.algorithms.s3fd.s3fd_fdm import S3FDFDM
from core.image.image import Image
from core.landmark_detection.algorithms.fan.fan_ldm import FANLDM

from enums import (
    DEVICE,
    FACE_DETECTION_ALGORITHM,
    LANDMARK_DETECTION_ALGORITHM,
)
from serializer.face_serializer import FaceSerializer
from utils import get_image_paths_from_dir

logger = logging.getLogger(__name__)


class ExtractorConfiguration:
    """Helper class for parsing arguments of the parser."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str = None,
        fda: Union[str, FACE_DETECTION_ALGORITHM] =
        FACE_DETECTION_ALGORITHM.S3FD,
        lda: Union[str, LANDMARK_DETECTION_ALGORITHM] =
        LANDMARK_DETECTION_ALGORITHM.FAN,
        quiet: bool = False,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        input_dir : str
            input directory with images
        output_dir : str, optional
            directory where face metadata will be saved, if no directory is
            passed, in the `input_dir` will be made new directory `metadata`
            and used for saving, by default None
        fda : Union[str, FACE_DETECTION_ALGORITHM], optional
            face detection algorithm, by default FACE_DETECTION_ALGORITHM.S3FD
        lda : Union[str, LANDMARK_DETECTION_ALGORITHM], optional
            landmark detection algorithm, by default
            LANDMARK_DETECTION_ALGORITHM.FAN
        quiet : bool, optional
            show progress of extraction or not, by default False
        """
        self.input_dir = input_dir
        if isinstance(fda, FACE_DETECTION_ALGORITHM):
            self.fda = fda
        else:
            self.fda = FACE_DETECTION_ALGORITHM[fda.upper()]
        if isinstance(lda, LANDMARK_DETECTION_ALGORITHM):
            self.lda = lda
        else:
            self.lda = LANDMARK_DETECTION_ALGORITHM[lda.upper()]
        if output_dir is None:
            output_dir = os.path.join(self.input_dir, 'metadata')
        self.output_dir = output_dir
        self.quiet = quiet

    def __str__(self):
        return 'Extractor configuration:\n' \
            + '------------------------\n' \
            + f'input directory: {self.input_dir}\n' \
            + f'face detection algorithm: {self.fda}\n' \
            + f'landmark detection algorithm: {self.lda}\n' \
            + f'output directory: {self.output_dir}'


class Extractor:

    def __init__(self, configuration: ExtractorConfiguration) -> None:
        """Constructor.

        Parameters
        ----------
        configuration : ExtractorConfiguration
            configuration object for extractor
        """
        self.input_dir = configuration.input_dir
        self.output_dir = configuration.output_dir

        fdm = configuration.fda
        if fdm == FACE_DETECTION_ALGORITHM.S3FD:
            self.fdm = S3FDFDM(DEVICE.CPU)
        elif fdm == FACE_DETECTION_ALGORITHM.FACEBOXES:
            self.fdm = FaceboxesFDM(DEVICE.CPU)

        ldm = configuration.lda
        if ldm == LANDMARK_DETECTION_ALGORITHM.FAN:
            self.ldm = FANLDM(DEVICE.CPU)

        self.verbose = not configuration.quiet

    def detect_faces(self, image: Image) -> List[Face]:
        """Initiates face detection process on the `image`. When face is
        detected, `Face` object is created and `bounding_box` property is set
        which bounds a face with two dots, upper left and lower right and they
        make a bounding rectangle.

        Parameters
        ----------
        image : Image
            image object with potential faces

        Returns
        -------
        List[Face]
            list of detected `Face` objects
        """
        faces = self.fdm.detect_faces(image)
        for f in faces:
            f.raw_image = image
        return faces

    def detect_landmarks(self, face: Face) -> None:
        """Initiates process of face landmark detection on the `face` object.
        After detection is done `landmarks` property is set on the `face`
        object.

        Note: face detection prior to this process has to be done in order to
        obtain bounding boxes of the potential faces.

        Parameters
        ----------
        face : Face
            face object containing bounding boxes
        """
        landmarks = self.ldm.detect_landmarks(face)
        face.landmarks = landmarks

    def run(self):
        """Initiates process of face and landmark extraction."""
        image_paths = get_image_paths_from_dir(self.input_dir)

        if image_paths:
            logger.info('Extraction started, please wait...')

            pbar = tqdm(
                image_paths,
                desc="Images done",
                disable=not self.verbose,
            )
            for path in pbar:
                image = Image.load(path)

                faces = self.detect_faces(image)

                for f in faces:
                    self.detect_landmarks(f)
                    FaceSerializer.save(f, self.output_dir)

            logger.info('Extraction process done.')
        else:
            logger.warning(f'No supported images in folder: {self.input_dir}.')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory where pictures with faces reside.',
    )
    parser.add_argument(
        '--fda',
        choices=[a.value for a in FACE_DETECTION_ALGORITHM],
        help='Available algorithms for face detection.',
        default=FACE_DETECTION_ALGORITHM.S3FD.value,
    )
    parser.add_argument(
        '--lda',
        choices=[a.value for a in LANDMARK_DETECTION_ALGORITHM],
        help='Available landmark detection algorithms.',
        default=LANDMARK_DETECTION_ALGORITHM.FAN.value,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory where Face objects whould be saved.'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="No ouput is shown when extraction process is running."
    )

    args = vars(parser.parse_args())

    ext_conf = ExtractorConfiguration(**args)

    ext = Extractor(ext_conf)

    ext.run()


if __name__ == '__main__':
    main()
