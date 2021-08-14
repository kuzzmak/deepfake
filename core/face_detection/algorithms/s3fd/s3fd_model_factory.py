import torch
import torch.nn as nn

from config import APP_CONFIG

from core.face_detection.algorithms.ModelFactory import ModelFactory
from core.face_detection.algorithms.s3fd.s3fd import build_s3fd

from enums import DEVICE


class S3FDModelFactory(ModelFactory):
    """Factory for S3FD face detection algorithm.
    """

    def build_model(device: DEVICE) -> nn.Module:
        net = build_s3fd('test')
        model_path = APP_CONFIG.app.core.face_detection.algorithms.s3fd.weight_path
        net.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device.value)
            )
        )
        net.eval()
        return net
