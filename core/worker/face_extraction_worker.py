import logging
from pathlib import Path
from typing import List, Optional, Union

import PyQt6.QtCore as qtc

from core.dictionary import Dictionary
from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from core.face_alignment.utils import get_face_mask
from core.face_detection.algorithms.faceboxes.faceboxes_fdm import FaceboxesFDM
from core.face_detection.algorithms.s3fd.s3fd_fdm import S3FDFDM
from core.image.image import Image
from core.landmark_detection.algorithms.fan.fan_ldm import FANLDM
from core.worker import Worker
from enums import (
    DEVICE,
    FACE_DETECTION_ALGORITHM,
    JOB_NAME,
    JOB_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)
from message.message import Messages
from serializer.face_serializer import FaceSerializer
from utils import get_image_paths_from_dir

logger = logging.getLogger(__name__)


class FaceExtractionWorker(Worker):
    """Worker for extracting faces from images along with face
    landmarks.

    Parameters
    ----------
    input_dir : Union[Path, str]
        directory with face images
    output_dir : Optional[Union[Path, str]], optional
        where to save extracted faces, if not provided, `metadata`
        directory will be made in `input_dir`, by default None
    fda : FACE_DETECTION_ALGORITHM, optional
        which face detection algorithm to use, by default
        FACE_DETECTION_ALGORITHM.S3FD
    device : DEVICE, optional
        which device to use for face detection and face landmarks
        detection, by default DEVICE.CPU
    message_worker_sig : Optional[qtc.pyqtSignal], optional
        signal to the message worker, by default None
    """

    def __init__(
        self,
        input_dir: Union[Path, str],
        output_dir: Optional[Union[Path, str]] = None,
        fda: FACE_DETECTION_ALGORITHM = FACE_DETECTION_ALGORITHM.S3FD,
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        if isinstance(input_dir, str):
            self._input_dir = Path(input_dir)
        else:
            self._input_dir = input_dir

        if output_dir is None:
            self._output_dir = self._input_dir / 'metadata'
        else:
            if isinstance(output_dir, str):
                self._output_dir = Path(output_dir)
            else:
                self._output_dir = output_dir

        if fda == FACE_DETECTION_ALGORITHM.S3FD:
            self._fdm = S3FDFDM(device)
        elif fda == FACE_DETECTION_ALGORITHM.FACEBOXES:
            self._fdm = FaceboxesFDM(device)

        self._ldm = FANLDM(device)
        self._device = device

    def _detect_faces(self, image: Image) -> List[Face]:
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
        faces = self._fdm.detect_faces(image)
        for f in faces:
            f.raw_image = image
        return faces

    def _detect_landmarks(self, face: Face) -> None:
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
        landmarks = self._ldm.detect_landmarks(face)
        face.landmarks = landmarks
        face.mask = get_face_mask(face.raw_image.data, landmarks.dots)

    def run_job(self) -> None:
        image_paths = get_image_paths_from_dir(self._input_dir)
        if not image_paths:
            logger.warning(
                f'No supported images in folder: {str(self._input_dir)}.'
            )
            return

        logger.info('Extraction started, please wait...')

        landmarks = Dictionary()
        alignments = Dictionary()

        self.running.emit()

        msg = Messages.CONFIGURE_WIDGET(
            SIGNAL_OWNER.FACE_EXTRACTION_WORKER,
            WIDGET.JOB_PROGRESS,
            'setMaximum',
            [len(image_paths)],
            JOB_NAME.FACE_EXTRACTION,
        )
        self.send_message(msg)

        for idx, i_p in enumerate(image_paths[:100]):
            image = Image.load(i_p)
            faces = self._detect_faces(image)

            for f in faces:
                self._detect_landmarks(f)
                FaceSerializer.save(f, self._output_dir)
                landmarks.add(f.name, f.landmarks.dots)
                FaceAligner.calculate_alignment(f)
                alignments.add(f.name, f.alignment)

            self.report_progress(
                SIGNAL_OWNER.FACE_EXTRACTION_WORKER,
                JOB_TYPE.FACE_EXTRACTION,
                idx,
                len(image_paths),
            )

        logger.debug('Saving landmarks.')
        landmarks.save(self._output_dir / 'landmarks.json')
        logger.debug('Landmarks saved.')
        logger.debug('Saving alignments.')
        alignments.save(self._output_dir / 'alignments.json')
        logger.debug('Alignments saved.')

        logger.info('Extraction process done.')
