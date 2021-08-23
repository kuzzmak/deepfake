import torch
import torch.nn as nn

from config import APP_CONFIG

from core.model_factory import ModelFactory

from enums import DEVICE

from utils import load_file_from_google_drive


class FANModelFactory(ModelFactory):
    """Model factory for FAN landmark detection model."""

    def build_model(device: DEVICE) -> nn.Module:
        model_id = APP_CONFIG.app.core.landmark_detection.algorithms.fan.gd_id
        model_path = load_file_from_google_drive(model_id, 'fan.zip')
        net = torch.jit.load(model_path)
        return net
