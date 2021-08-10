from enums import DEVICE
from easydict import EasyDict

import torch

devices = [DEVICE.CPU]
if torch.cuda.device_count() > 0:
    devices.append(DEVICE.CUDA)


_C = EasyDict()
APP_CONFIG = _C

_C.app = EasyDict()
_C.app.input_faces_directory = 'C:\\Users\\tonkec\\Documents\\deepfake\\data\\input_faces_data'
_C.app.output_faces_directory = 'C:\\Users\\tonkec\\Documents\\deepfake\\data\\output_faces_data'
_C.app.s3fd_model_path = 'C:\\Users\\tonkec\\Documents\\deepfake\\data\\weights\\s3fd\\s3fd.pth'

_C.app.console = EasyDict()
_C.app.console.font_name = 'Consolas'
_C.app.console.text_size = 10

_C.app.window = EasyDict()
_C.app.window.preferred_width = 1280
_C.app.window.preferred_height = 720

_C.app.core = EasyDict()
_C.app.core.devices = devices
_C.app.core.selected_device = DEVICE.CPU

_C.app.core.face_detection = EasyDict()

_C.app.core.face_detection.algorithms = EasyDict()

_C.app.core.face_detection.algorithms.s3fd = EasyDict()
_C.app.core.face_detection.algorithms.s3fd.weight_path = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\weights\\s3fd\\s3fd.pth"


_C.app.gui = EasyDict()

_C.app.gui.video_widget = EasyDict()
_C.app.gui.video_widget.video_aspect_ratio = 16. / 9
