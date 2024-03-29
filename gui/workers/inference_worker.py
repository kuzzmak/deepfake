import logging

import numpy as np
import cv2 as cv
import PyQt6.QtCore as qtc
import torch
from torchvision import transforms

from configs.app_config import APP_CONFIG
from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from core.face_alignment.utils import get_face_mask
from core.face_detection.algorithms.faceboxes.faceboxes_fdm import FaceboxesFDM
from core.face_detection.algorithms.s3fd.s3fd_fdm import S3FDFDM
from core.image.image import Image
from core.landmark_detection.algorithms.fan.fan_ldm import FANLDM
from core.model.original_ae import OriginalAE
from enums import DEVICE, FACE_DETECTION_ALGORITHM, MASK_DIM
from utils import tensor_to_np_image


logger = logging.getLogger(__name__)


class InferenceWorker(qtc.QObject):

    image_sig = qtc.pyqtSignal(Image)
    model_sig = qtc.pyqtSignal(str)
    device_sig = qtc.pyqtSignal(DEVICE)
    algorithm_sig = qtc.pyqtSignal(FACE_DETECTION_ALGORITHM)
    inference_result = qtc.pyqtSignal(torch.Tensor, torch.Tensor, torch.Tensor)
    inference_started = qtc.pyqtSignal()
    inference_finished = qtc.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._ldm = None
        self._fdm = None
        self._model_path = None
        self._face_detection_algorithm = \
            APP_CONFIG.app.core.face_detection.algorithms.default
        self._device = DEVICE.CPU
        self.image_sig.connect(self.run)
        self.model_sig.connect(self._load_model)
        self.device_sig.connect(self._device_changed)
        self.algorithm_sig.connect(self._algorithm_changed)

    @qtc.pyqtSlot(FACE_DETECTION_ALGORITHM)
    def _algorithm_changed(self, algorithm: FACE_DETECTION_ALGORITHM) -> None:
        self._face_detection_algorithm = algorithm
        self._load_face_detection_model()

    @qtc.pyqtSlot(DEVICE)
    def _device_changed(self, device: DEVICE) -> None:
        logger.debug(
            f'Device changed from {self._device.value} to {device.value}.'
        )
        self._device = device
        self._load_face_detection_model()
        self._load_landmark_detection_model()
        if self._model_path is not None:
            self._load_model(self._model_path)

    @qtc.pyqtSlot(str)
    def _load_model(self, model_path: str):
        self._model_path = model_path
        logger.info(f'Loading deepfake model on device {self._device.value}.')
        self._model = OriginalAE((3, 64, 64))
        weights = torch.load(model_path, map_location=self._device.value)
        self._model.load_state_dict(weights)
        self._model.to(self._device.value)
        self._model.eval()
        logger.info('Model loaded.')

    def _load_landmark_detection_model(self) -> None:
        logger.info(
            f'Loading landmark detection model on device {self._device.value}.'
        )
        self._ldm = FANLDM(self._device)
        logger.info('Landmark detection model loaded.')

    def _load_face_detection_model(self) -> None:
        logger.info(
            'Loading face detection model ' +
            f'({self._face_detection_algorithm.value}).'
        )
        if self._face_detection_algorithm == FACE_DETECTION_ALGORITHM.S3FD:
            self._fdm = S3FDFDM(self._device)
        else:
            self._fdm = FaceboxesFDM(self._device)
        logger.info('Face detection model loaded.')

    @staticmethod
    def _merge(face: Face, prediction: torch.Tensor) -> np.ndarray:
        pred = tensor_to_np_image(prediction).astype(np.uint8)
        pred = cv.cvtColor(pred, cv.COLOR_RGB2BGR)
        pred = cv.resize(pred, (64, 64), interpolation=cv.INTER_CUBIC)

        mask = face.get_mask(mask_dim=MASK_DIM.THREE)
        pred = pred * mask

        negative_mask = 1 - mask
        clone = face.aligned_image.copy()
        clone = clone * negative_mask + pred

        mask = face.get_mask(fill_values=255)
        max_region = np.argwhere(mask == 255)
        min_y, min_x = max_region.min(axis=0)
        max_y, max_x = max_region.max(axis=0)
        len_x = max_x - min_x
        len_y = max_y - min_y
        mask_y = min_x + len_x // 2
        mask_x = min_y + len_y // 2

        return cv.seamlessClone(
            clone.astype(np.uint8),
            face.aligned_image,
            mask.astype(np.uint8),
            (mask_y, mask_x),
            cv.NORMAL_CLONE,
        )

    @qtc.pyqtSlot(Image)
    def run(self, image: Image) -> None:
        if self._fdm is None:
            self._load_face_detection_model()
        if self._ldm is None:
            self._load_landmark_detection_model()
        if self._model is None:
            logger.error('Can not run inference, model was not loaded.')
            return

        self.inference_started.emit()
        logger.debug('Detecting faces...')
        faces = self._fdm.detect_faces(image)
        logger.debug(f'Detected {len(faces)} face(s).')
        for i, face in enumerate(faces):
            face.raw_image = image
            logger.debug(f'Detecting landmarks for face {i}...')
            landmarks = self._ldm.detect_landmarks(face)
            logger.debug(f'Landmark detection done for face {i}.')
            face.landmarks = landmarks
            face.mask = get_face_mask(face.raw_image.data, landmarks.dots)
            FaceAligner.align_face(face, 64)

            img_ten = transforms.ToTensor()(face.aligned_image)
            img_ten = img_ten.unsqueeze(0)
            img_ten = img_ten.to(self._device.value)
            y_pred_A_A, _, y_pred_A_B, _ = self._model(img_ten)

            prediction = y_pred_A_B.squeeze(0)

            clone = InferenceWorker._merge(face, prediction)

            self.inference_result.emit(
                transforms.ToTensor()(face.aligned_image),
                prediction,
                transforms.ToTensor()(clone),
            )
        self.inference_finished.emit()
