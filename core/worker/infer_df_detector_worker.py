import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image
import PyQt6.QtCore as qtc
import torch
from torchvision.transforms import transforms

from configs.mri_gan_config import MRIGANConfig
from core.df_detection.mri_gan.data_utils.face_detection import get_face_detector_model
from core.df_detection.mri_gan.deep_fake_detect.DeepFakeDetectModel import \
    DeepFakeDetectModel
from core.df_detection.mri_gan.deep_fake_detect.utils import ENCODER_PARAMS, get_predictions, get_probability
from core.worker import ContinuousWorker
from enums import DEVICE, JOB_DATA_KEY
from utils import prepare_path
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
        model_path: Union[Path, str],
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._device = device
        self._model_path = prepare_path(model_path)
        self._model = None

    def _load_model(self) -> None:
        if self._model_path is None:
            logger.warning(
                'Model path is not valid, please check it and ' +
                f'try again. Current model path: {str(self._model_path)}.'
            )
            return

        logger.debug(f'Loading df detector model to: {self._device.value}.')
        model_dict = torch.load(
            self._model_path,
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

    def _detect_face(self, image: Image.Image) -> Union[Image.Image, None]:
        detector = get_face_detector_model()
        boxes, _ = detector.detect(image)
        if boxes is None:
            return None
        boxes = [b.tolist() if b is not None else None for b in boxes]
        xmin, ymin, xmax, ymax = [int(b) for b in boxes[0]]
        w = xmax - xmin
        h = ymax - ymin
        buf = 0.10
        p_h = int(h * buf)
        p_w = int(w * buf)
        return image.crop(
            (
                max(ymin - p_h, 0),
                max(xmin - p_w, 0),
                ymax + p_h,
                xmax + p_w,
            )
        )

    def run_job(self) -> None:
        if self._model is None:
            self._load_model()
            if self._model is None:
                return

        self.running.emit()

        path = self._current_job.data.get(JOB_DATA_KEY.IMAGE_PATH, None)
        if path is None:
            logger.error(
                f'Key {JOB_DATA_KEY.IMAGE_PATH.value} must be present.'
            )
            return
        path = prepare_path(path)
        if path is None:
            logger.error(f'Unable to parse image path {str(path)}.')
            return

        image = Image.open(path)

        face = self._detect_face(image)

        if face is None:
            logger.warning(
                f'Was not able to detect face on image {str(path)}.'
            )
            return

        encoder_name = MRIGANConfig \
            .get_instance() \
            .get_default_cnn_encoder_name()
        image_size = ENCODER_PARAMS[encoder_name]['imsize']
        face = _image_transforms(image_size)(face)
        face = face.unsqueeze(0)

        print('size', face.size())

        with torch.no_grad():
            output = self._model(face)

        output = output.squeeze(0)
        preds = get_predictions(output).to('cpu').detach().numpy()
        class_probability = get_probability(output).to('cpu').detach().numpy()

        print(preds, class_probability)
