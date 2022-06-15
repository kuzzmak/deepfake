import logging
import os
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import numpy as np
import PyQt6.QtCore as qtc
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.aligner import Aligner, AlignerConfiguration
from core.dataset.simple_df_dataset import SimpleDFDataset
from core.model.original_ae import OriginalAE
from core.worker import Worker, FramesExtractionWorker, FaceExtractionWorker
from enums import DEVICE
from utils import prepare_path, tensor_to_np_image
from variables import DATA_ROOT, SUPPORTED_VIDEO_EXTS

logger = logging.getLogger(__name__)


def clean_dir(directory: Path) -> None:
    [os.remove(f) for f in list(directory.glob('*.*'))]


class InferDFModelWorker(Worker):

    def __init__(
        self,
        data_path: Union[str, Path],
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._data_path = prepare_path(data_path)
        self._device = device

    @qtc.pyqtSlot(str)
    def _load_model(self, model_path: str) -> OriginalAE:
        logger.info(f'Loading deepfake model on device {self._device.value}.')
        model = OriginalAE((3, 64, 64))
        weights = torch.load(model_path, map_location=self._device.value)
        model.load_state_dict(weights)
        model.to(self._device.value)
        model.eval()
        logger.info('Model loaded.')
        return model

    def run_job(self) -> None:
        if self._data_path is None:
            logger.error('Invalid path provided.')
            return
        if not self._data_path.exists():
            logger.error('Provided data path doesn\'t exist.')
            return
        if self._data_path.suffix not in SUPPORTED_VIDEO_EXTS:
            logger.error('Provided file type not supported. ')
            return

        temp_dir = DATA_ROOT / 'temp'
        if not temp_dir.exists():
            logger.debug(f'Making temp directory {str(temp_dir)}.')
            temp_dir.mkdir()

        frames_dir = temp_dir / self._data_path.stem
        if not frames_dir.exists():
            logger.debug(f'Making directory for frames {str(frames_dir)}.')
            frames_dir.mkdir()
        # else:
        #     clean_dir(frames_dir)

        self.running.emit()

        # extract every frame from the video
        # worker = FramesExtractionWorker(
        #     self._data_path,
        #     frames_dir,
        #     1,
        #     self.message_worker_sig,
        # )
        # worker.run_job()

        # worker = FaceExtractionWorker(
        #     input_dir=frames_dir,
        #     device=self._device,
        #     message_worker_sig=self.message_worker_sig,
        # )
        # worker.run_job()

        metadata_path = frames_dir / 'metadata'

        # a_c = AlignerConfiguration(
        #     metadata_path,
        #     64,
        #     self.message_worker_sig,
        # )
        # aligner = Aligner(a_c)
        # aligner.align_landmarks()

        model_path = r'C:\Users\tonkec\Desktop\best_model_300.pt'
        model = self._load_model(model_path)

        predicted_dir = temp_dir / 'predicted'
        if not predicted_dir.exists():
            predicted_dir.mkdir()
        else:
            clean_dir(predicted_dir)

        dataset = SimpleDFDataset(metadata_path, 64)
        dataloader = DataLoader(dataset, 32, pin_memory=True, num_workers=4)
        img_counter = 0
        for data in tqdm(dataloader):
            with torch.no_grad():
                pred_A, _, pred_B, _ = model(data.to(self._device.value))
            images = [tensor_to_np_image(im) for im in pred_A]
            for im in images:
                save_path = Path(predicted_dir) / f'pred_{img_counter}.jpg'
                img_counter += 1
                print(f'saving {str(save_path)}')
                img = im.astype(np.uint8)
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                # pred = cv.resize(pred, (64, 64), interpolation=cv.INTER_CUBIC)
                ok = cv.imwrite(str(save_path), img)
                if not ok:
                    logger.error(f'Unable to save image {str(save_path)}.')

        logger.info('Deepfake creation done.')
