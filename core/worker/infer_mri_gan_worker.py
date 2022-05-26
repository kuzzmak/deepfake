import logging
from typing import Optional

import PyQt6.QtCore as qtc
import torch
from torchvision.transforms import transforms

from config import APP_CONFIG
from configs.mri_gan_config import MRIGANConfig
from core.df_detection.mri_gan.mri_gan.model import GeneratorUNet, get_MRI_GAN
from core.worker import ContinuousWorker
from enums import DEVICE
from utils import load_file_from_google_drive

logger = logging.getLogger(__name__)


class InferMRIGANWorker(ContinuousWorker):

    def __init__(
        self,
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._device = device
        self._model = None

    def _load_model(self) -> None:
        model_id = APP_CONFIG \
            .app \
            .core \
            .df_detection \
            .models \
            .mri_gan \
            .submodels \
            .mri_gan \
            .gd_id
        model_path = load_file_from_google_drive(model_id, 'mri_gan.chkpt')

        logger.debug(f'Loading MRI GAN model to: {self._device.value}.')
        self._model = GeneratorUNet()
        checkpoint = torch.load(
            model_path,
            map_location=torch.device(self._device.value),
        )
        self._model.load_state_dict(checkpoint['generator_state_dict'])
        self._model = self._model.to(self._device.value)
        self._model.eval()
        logger.debug('Model loading finished.')

    def run_job(self) -> None:
        logger.info('MRI prediction for image started.')
        if self._model is None:
            self._load_model()

        im_size = MRIGANConfig \
            .get_instance() \
            .get_mri_gan_model_params()['imsize']
        transforms_ = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
