import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from enums import DEVICE, FACE_DETECTION_ALGORITHM


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
    default: FACE_DETECTION_ALGORITHM
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
class _ImageViewerSorter:
    images_per_page_options: List[int]


@dataclass
class _Widgets:
    video_widget: _VideoWidget
    console: _Console
    image_viewer_sorter: _ImageViewerSorter


@dataclass
class _Gui:
    window: _Window
    widgets: _Widgets


@dataclass
class _GoogleImagesScraper:
    default_save_directory: Path
    suggested_search_depth: int
    default_page_limit: int


@dataclass
class _Resources:
    face_example_path: str


@dataclass
class _App:
    core: _Core
    gui: _Gui
    google_images_scraper: _GoogleImagesScraper
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

        _default_face_detection_algorithm = FACE_DETECTION_ALGORITHM[
            _face_detection_algorithms['default'].upper()
        ]

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

        _image_viewer_sorter = _widgets['image_viewer_sorter']
        images_per_page_options = _image_viewer_sorter[
            'images_per_page_options'
        ]

        font_name = _console['font_name']
        text_size = _console['text_size']

        _google_images_scraper = _app['google_images_scraper']
        default_save_directory = _google_images_scraper[
            'default_save_directory'
        ]
        default_save_directory = Path(__file__) \
            .resolve() \
            .parent / default_save_directory
        suggested_search_depth = _google_images_scraper[
            'suggested_search_depth'
        ]
        default_page_limit = _google_images_scraper[
            'default_page_limit'
        ]

        _resources = _app['resources']
        face_example_path = _resources['face_example_path']

        devices = [DEVICE.CPU]
        if torch.cuda.device_count() > 0:
            devices.append(DEVICE.CUDA)

        return Config(
            _App(
                _Core(
                    _FaceDetection(
                        _FaceDetectionAlgorithms(
                            _default_face_detection_algorithm,
                            _S3FD(s3fd_gd_id),
                            _FaceBoxes(faceboxes_gd_id),
                        )
                    ),
                    _LandmarkDetection(
                        _LandmarkDetectionAlgorithms(_FAN(fan_gd_id))
                    ),
                    devices,
                    selected_device,
                ),
                _Gui(
                    _Window(preferred_width, preferred_height),
                    _Widgets(
                        _VideoWidget(video_aspect_ratio),
                        _Console(font_name, text_size),
                        _ImageViewerSorter(images_per_page_options)
                    ),
                ),
                _GoogleImagesScraper(
                    default_save_directory,
                    suggested_search_depth,
                    default_page_limit,
                ),
                _Resources(face_example_path),
            )
        )


APP_CONFIG = _load_config()


def refresh_config():
    global APP_CONFIG
    APP_CONFIG = _load_config()
