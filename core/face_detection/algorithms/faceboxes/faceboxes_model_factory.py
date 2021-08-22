import torch
import torch.nn as nn

from config import APP_CONFIG

from core.face_detection.algorithms.faceboxes.faceboxes import FaceBoxes
from core.model_factory import ModelFactory

from enums import DEVICE

from utils import load_file_from_google_drive


class FaceboxesModelFactory(ModelFactory):

    def build_model(device: DEVICE) -> nn.Module:
        net = FaceBoxes()
        model_id = APP_CONFIG.app.core.face_detection.algorithms.faceboxes.gd_id
        dict_path = load_file_from_google_drive(model_id, 'faceboxes.pth')
        net.load_state_dict(
            torch.load(
                dict_path,
                map_location=torch.device(device.value)
            )
        )
        return net
