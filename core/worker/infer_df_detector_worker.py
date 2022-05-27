import logging
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import PyQt6.QtCore as qtc
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import APP_CONFIG
from configs.mri_gan_config import MRIGANConfig
from core.df_detection.mri_gan.data_utils.datasets import SimpleImageFolder
from core.df_detection.mri_gan.data_utils.face_detection import (
    crop_faces_from_video,
    extract_landmarks_from_video,
)
from core.df_detection.mri_gan.deep_fake_detect.DeepFakeDetectModel import \
    DeepFakeDetectModel
from core.df_detection.mri_gan.deep_fake_detect.utils import (
    ENCODER_PARAMS,
    get_predictions,
    get_probability,
    pred_strategy,
)
from core.worker import ContinuousWorker, PredictMRIWorker
from enums import DEVICE, JOB_DATA_KEY, MRI_GAN_DATASET, OUTPUT_KEYS
from utils import load_file_from_google_drive, prepare_path
from variables import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def _image_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class InferDFDetectorWorker(ContinuousWorker):

    def __init__(
        self,
        df_detection_model: MRI_GAN_DATASET,
        fake_threshold: float,
        fake_fraction: float,
        batch_size: int,
        num_workers: int,
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._fake_threshold = fake_threshold
        self._fake_fraction = fake_fraction
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._df_detection_model = df_detection_model
        self._device = device
        self._model = None

    def _load_model(self) -> None:
        if self._df_detection_model == MRI_GAN_DATASET.MRI:
            model_id = APP_CONFIG \
                .app \
                .core \
                .df_detection \
                .models \
                .mri_gan \
                .submodels \
                .mri_gan_df_detector \
                .gd_id
        elif self._df_detection_model == MRI_GAN_DATASET.PLAIN:
            model_id = APP_CONFIG \
                .app \
                .core \
                .df_detection \
                .models \
                .mri_gan \
                .submodels \
                .plain_df_detector \
                .gd_id
        else:
            logger.error('Unsupported model for MRI GAN deepfake detector.')
            return

        model_path = load_file_from_google_drive(
            model_id,
            f'{self._df_detection_model.value.lower()}.chkpt',
        )

        logger.debug(f'Loading df detector model to: {self._device.value}.')
        model_dict = torch.load(
            model_path,
            map_location=torch.device(self._device.value),
        )
        model_params = model_dict['model_params']
        self._model = DeepFakeDetectModel(
            model_params['imsize'],
            model_params['encoder_name'],
        )
        self._model.load_state_dict(model_dict['model_state_dict'], False)
        self._model = self._model.to(self._device.value)
        self._model.eval()
        logger.debug('Model loaded.')

    def run_job(self) -> None:
        logger.info('Prediction for video started.')
        if self._model is None:
            self._load_model()
            if self._model is None:
                return

        self.running.emit()

        path = self._current_job.data.get(JOB_DATA_KEY.FILE_PATH, None)
        if path is None:
            logger.error(
                f'Key {JOB_DATA_KEY.FILE_PATH.value} must be present.'
            )
            return
        path = prepare_path(path)
        if path is None:
            logger.error(f'Unable to parse file path {str(path)}.')
            return

        root_dir = tempfile.TemporaryDirectory()
        root_path = Path(root_dir.name)
        plain_faces_data_dir = root_path / 'plain_frames'
        plain_faces_data_dir.mkdir(exist_ok=True)

        logger.debug(f'Extracting landmarks from video {str(path)}.')
        extract_landmarks_from_video(
            path,
            root_path,
            overwrite=True,
            inference=True,
        )
        logger.debug('Landmark extraction done.')

        logger.debug(f'Cropping faces from video {str(path)}.')
        crop_faces_from_video(
            path,
            root_path,
            plain_faces_data_dir,
            overwrite=True,
            inference=True,
        )
        logger.debug('Face cropping finished.')

        if self._df_detection_model == MRI_GAN_DATASET.PLAIN:
            frames_path = plain_faces_data_dir
        else:
            frames_path = root_path / 'mri'
            frames_path.mkdir(exist_ok=True)
            logger.debug(f'Predicting MRI for video {str(path)}.')
            PredictMRIWorker._predict_mri_using_MRI_GAN(
                frames_path,
                plain_faces_data_dir / path.stem,
                8,
                True,
                True,
                True,
            )
            logger.debug('MRI prediction done.')

        encoder_name = MRIGANConfig \
            .get_instance() \
            .get_default_cnn_encoder_name()
        image_size = ENCODER_PARAMS[encoder_name]['imsize']

        test_transform = _image_transforms(image_size)
        data_path = plain_faces_data_dir / path.stem
        logger.debug('Constructing simple dataset.')
        test_dataset = SimpleImageFolder(data_path, test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        logger.debug(f'Dataset constructed. Total {len(test_dataset)} frames.')
        probabilities = []
        with torch.no_grad():
            for samples in test_loader:
                frames = samples.to(self._device.value)
                output = self._model(frames)
                predicted = get_predictions(output) \
                    .to('cpu') \
                    .detach() \
                    .numpy()
                class_probability = get_probability(output) \
                    .to('cpu') \
                    .detach() \
                    .numpy()
                if len(predicted) > 1:
                    probabilities.extend(class_probability.squeeze())
                else:
                    probabilities.append(class_probability.squeeze())

            total_number_frames = len(probabilities)
            probabilities = np.array(probabilities)

            fake_frames_high_prob = probabilities[
                probabilities >= self._fake_threshold
            ]
            number_fake_frames = len(fake_frames_high_prob)
            if number_fake_frames == 0:
                fake_prob = 0
            else:
                fake_prob = round(
                    sum(fake_frames_high_prob) /
                    number_fake_frames,
                    4,
                )

            real_frames_high_prob = probabilities[
                probabilities < self._fake_threshold
            ]
            number_real_frames = len(real_frames_high_prob)
            if number_real_frames == 0:
                real_prob = 0
            else:
                real_prob = 1 - round(
                    sum(real_frames_high_prob) / number_real_frames,
                    4,
                )

            pred = pred_strategy(
                number_fake_frames,
                number_real_frames,
                total_number_frames,
                fake_fraction=self._fake_fraction,
            )

            print(fake_prob, real_prob, pred)

            self.output.emit({
                OUTPUT_KEYS.FAKE_PROB: fake_prob,
                OUTPUT_KEYS.REAL_PROB: real_prob,
                OUTPUT_KEYS.PREDICTION: pred,
            })

        root_dir.cleanup()

        logger.info('Prediction done.')
