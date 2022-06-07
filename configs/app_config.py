import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from enums import DEVICE, FACE_DETECTION_ALGORITHM
from variables import APP_CONFIG_PATH


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
class _Model:
    id: str
    gd_id: str
    name: str


@dataclass
class _DFDetectionSubmodels:
    plain_df_detector: _Model
    mri_gan_df_detector: _Model
    mri_gan: _Model


@dataclass
class _MRIGANDFDetectionModel(_Model):
    submodels: _DFDetectionSubmodels


@dataclass
class _DFDetectionModels:
    mri_gan: _MRIGANDFDetectionModel
    meso_net: _Model


@dataclass
class _DFDetection:
    models: _DFDetectionModels


@dataclass
class _Core:
    face_detection: _FaceDetection
    landmark_detection: _LandmarkDetection
    df_detection: _DFDetection
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
class AppConfig:
    app: _App


def _load_config():
    conf_path = APP_CONFIG_PATH
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

        ##############
        # DF DETECTION
        ##############
        _df_detection = _core['df_detection']

        _df_detection_models = _df_detection['models']

        _meso_net = _Model(
            **_df_detection_models['meso_net']
        )

        _mri_gan = _df_detection_models['mri_gan']
        _submodels = _mri_gan['submodels']
        _plain_df_detector_submodel = _Model(**_submodels['plain_df_detector'])
        _mri_gan_df_detector_submodel = _Model(
            **_submodels['mri_gan_df_detector']
        )
        _mri_gan_submodel = _Model(**_submodels['mri_gan'])

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

        return AppConfig(
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
                    _DFDetection(
                        _DFDetectionModels(
                            _MRIGANDFDetectionModel(
                                _mri_gan['id'],
                                _mri_gan['gd_id'],
                                _mri_gan['name'],
                                _DFDetectionSubmodels(
                                    _plain_df_detector_submodel,
                                    _mri_gan_df_detector_submodel,
                                    _mri_gan_submodel,
                                )
                            ),
                            _meso_net,
                        )
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
