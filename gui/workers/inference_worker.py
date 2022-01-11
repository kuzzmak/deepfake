import logging

import cv2 as cv
import PyQt5.QtCore as qtc
import torch

from core.image.image import Image
from core.model.model import DeepfakeModel
from core.model.original_ae import OriginalAE
from enums import DEVICE


logger = logging.getLogger(__name__)


class InferenceWorker(qtc.QObject):

    image_sig = qtc.pyqtSignal(Image)
    model_sig = qtc.pyqtSignal(str)
    device_sig = qtc.pyqtSignal(DEVICE)

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = DEVICE.CPU
        self.image_sig.connect(self.run)
        self.model_sig.connect(self._load_model)
        self.device_sig.connect(self._device_changed)

    @qtc.pyqtSlot(DEVICE)
    def _device_changed(self, device: DEVICE) -> None:
        self._device = device
        print('new device', device)

    @qtc.pyqtSlot(str)
    def _load_model(self, model_path: str) -> DeepfakeModel:
        logger.info('Loading model.')
        self._model = OriginalAE((3, 64, 64))
        weights = torch.load(model_path, map_location=self._device.value)
        self._model.load_state_dict(weights)
        self._model.to(self._device.value)
        self._model.eval()
        logger.info('Model loaded.')

    @qtc.pyqtSlot(Image)
    def run(self, image: Image) -> None:
        cv.imshow('image', image.data)
        cv.waitKey()
