import json
from dataclasses import dataclass
from typing import List

import torch

from enums import DEVICE


@dataclass
class _S3FD:
    gd_id: str


@dataclass
class _FaceBoxes:
    gd_id: str


@dataclass
class _MTCNN:
    gd_id: str


@dataclass
class _FaceDetectionAlgorithms:
    s3fd: _S3FD
    faceboxes: _FaceBoxes
    # mtcnn: _MTCNN


@dataclass
class _FaceDetection:
    algorithms: _FaceDetectionAlgorithms


@dataclass
class _FAN:
    gd_id: str


@dataclass
class _LandmarkDetectionAlgorithms:
    fan: _FAN


@dataclass
class _LandmarkDetection:
    algorithms: _LandmarkDetectionAlgorithms


@dataclass
class _Core:
    face_detection: _FaceDetection
    landmark_detection: _LandmarkDetection
    devices: List[DEVICE]
    selected_device: DEVICE = DEVICE.CPU


@dataclass
class _Window:
    preferred_width: int
    preferred_height: int


@dataclass
class _VideoWidget:
    video_aspect_ratio: float


@dataclass
class _Console:
    font_name: str
    text_size: int


@dataclass
class _Widgets:
    video_widget: _VideoWidget
    console: _Console


@dataclass
class _Gui:
    window: _Window
    widgets: _Widgets


@dataclass
class _Resources:
    face_example_path: str


@dataclass
class _App:
    input_faces_directory: str
    output_faces_directory: str
    core: _Core
    gui: _Gui
    resources: _Resources


@dataclass
class Config:
    app: _App


def _load_config():
    conf_path = 'config.json'
    with open(conf_path) as f:
        conf = f.read()
        conf = json.loads(conf)

        _app = conf['app']

        input_faces_directory = _app['input_faces_directory']
        output_faces_directory = _app['output_faces_directory']

        _core = _app['core']
        selected_device = _core['selected_device']
        if selected_device == 'cpu':
            selected_device = DEVICE.CPU
        else:
            selected_device = DEVICE.CUDA

        _face_detection = _core['face_detection']

        _face_detection_algorithms = _face_detection['algorithms']

        _s3fd = _face_detection_algorithms['s3fd']
        s3fd_gd_id = _s3fd['gd_id']

        _faceboxes = _face_detection_algorithms['faceboxes']
        faceboxes_gd_id = _faceboxes['gd_id']

        _landmark_detection = _core['landmark_detection']

        _landmark_detection_algorithms = _landmark_detection['algorithms']

        _fan = _landmark_detection_algorithms['fan']
        fan_gd_id = _fan['gd_id']

        _gui = _app['gui']

        _window = _gui['window']
        preferred_height = _window['preferred_height']
        preferred_width = _window['preferred_width']

        _widgets = _gui['widgets']

        _video_widget = _widgets['video_widget']
        video_aspect_ratio = _video_widget['video_aspect_ratio']

        _console = _widgets['console']

        font_name = _console['font_name']
        text_size = _console['text_size']

        _resources = _app['resources']
        face_example_path = _resources['face_example_path']

        devices = [DEVICE.CPU]
        if torch.cuda.device_count() > 0:
            devices.append(DEVICE.CUDA)

        conf = Config(
            _App(
                input_faces_directory,
                output_faces_directory,
                _Core(
                    _FaceDetection(
                        _FaceDetectionAlgorithms(
                            _S3FD(s3fd_gd_id),
                            _FaceBoxes(faceboxes_gd_id),
                        )
                    ),
                    _LandmarkDetection(
                        _LandmarkDetectionAlgorithms(
                            _FAN(fan_gd_id)
                        )
                    ),
                    devices,
                    selected_device,
                ),
                _Gui(
                    _Window(
                        preferred_width,
                        preferred_height,
                    ),
                    _Widgets(
                        _VideoWidget(
                            video_aspect_ratio,
                        ),
                        _Console(
                            font_name,
                            text_size,
                        )
                    ),
                ),
                _Resources(face_example_path),
            )
        )

        return conf


APP_CONFIG = _load_config()


def refresh_config():
    global APP_CONFIG
    APP_CONFIG = _load_config()
