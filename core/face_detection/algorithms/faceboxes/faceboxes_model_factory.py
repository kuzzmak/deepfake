import torch
import torch.nn as nn

from config import APP_CONFIG

from core.face_detection.algorithms.faceboxes.faceboxes import FaceBoxes
from core.face_detection.algorithms.ModelFactory import ModelFactory

from enums import DEVICE


class FaceboxesModelFactory(ModelFactory):

    def build_model(device: DEVICE) -> nn.Module:
        net = FaceBoxes()
        model_path = APP_CONFIG.app.core.face_detection.algorithms.faceboxes.weight_path
        net.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device.value)
            )
        )
        net.eval()
        return net
