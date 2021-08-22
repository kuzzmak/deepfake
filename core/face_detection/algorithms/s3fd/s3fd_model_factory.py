import torch
import torch.nn as nn

from config import APP_CONFIG

from core.model_factory import ModelFactory
from core.face_detection.algorithms.s3fd.s3fd import build_s3fd

from enums import DEVICE

from utils import load_file_from_google_drive


class S3FDModelFactory(ModelFactory):
    """Factory for S3FD face detection algorithm.
    """

    def build_model(device: DEVICE) -> nn.Module:
        net = build_s3fd('test')
        model_id = APP_CONFIG.app.core.face_detection.algorithms.s3fd.gd_id
        dict_path = load_file_from_google_drive(model_id, 's3fd.pth')
        net.load_state_dict(
            torch.load(
                dict_path,
                map_location=torch.device(device.value)
            )
        )
        return net
