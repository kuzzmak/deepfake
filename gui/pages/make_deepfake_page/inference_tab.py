import logging
from pathlib import Path
from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
import torch

from config import APP_CONFIG
from core.image.image import Image
from gui.workers.inference_worker import InferenceWorker
from enums import DEVICE, FACE_DETECTION_ALGORITHM, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import HWidget, VWidget, VerticalSpacer
from gui.widgets.preview.new_preview import Preview

logger = logging.getLogger(__name__)


class InferenceTab(BaseWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
    ):
        super().__init__(signals)
        # self._start_inference_thread()
        self._widgets_to_disable_on_inference = []
        self._threads = []
        self._last_model_folder = None
        self._last_image_folder = None
        self._image_path = None
        # self._inference_worker.inference_result.connect(self._inference_result)
        # self._inference_worker.inference_started.connect(
        #     self._on_inference_start
        # )
        # self._inference_worker.inference_finished.connect(
        #     self._on_inference_finished
        # )
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QHBoxLayout()
        self.setLayout(layout)
        left_part = VWidget()
        layout.addWidget(left_part)
        left_part.setMaximumWidth(300)

        model_gb = qwt.QGroupBox()
        self._widgets_to_disable_on_inference.append(model_gb)
        left_part.layout().addWidget(model_gb)
        model_gb.setTitle('Model selection')
        model_gb_layout = qwt.QHBoxLayout(model_gb)
        model_select_btn = qwt.QPushButton(text='select')
        model_gb_layout.addWidget(model_select_btn)
        model_select_btn.clicked.connect(self._load_model)

        image_gb = qwt.QGroupBox()
        self._widgets_to_disable_on_inference.append(image_gb)
        left_part.layout().addWidget(image_gb)
        image_gb.setTitle('Image selection')
        image_gb_layout = qwt.QHBoxLayout(image_gb)
        image_select_btn = qwt.QPushButton(text='select')
        image_gb_layout.addWidget(image_select_btn)
        image_select_btn.clicked.connect(self._load_image)

        algorithm_gb = qwt.QGroupBox(
            title='Available face detection algorithms'
        )
        self._widgets_to_disable_on_inference.append(algorithm_gb)
        left_part.layout().addWidget(algorithm_gb)
        algorithm_gb_layout = qwt.QHBoxLayout(algorithm_gb)
        algorithm_row = HWidget()
        algorithm_row.layout().setContentsMargins(0, 0, 0, 0)
        algorithm_gb_layout.addWidget(algorithm_row)
        self.algorithm_bg = qwt.QButtonGroup(algorithm_row)
        for alg in FACE_DETECTION_ALGORITHM:
            btn = qwt.QRadioButton(alg.value, algorithm_gb)
            if alg == APP_CONFIG.app.core.face_detection.algorithms.default:
                btn.setChecked(True)
            btn.toggled.connect(self._face_detection_algorithm_changed)
            algorithm_row.layout().addWidget(btn)
            self.algorithm_bg.addButton(btn)

        device_gb = qwt.QGroupBox()
        self._widgets_to_disable_on_inference.append(device_gb)
        left_part.layout().addWidget(device_gb)
        device_gb.setTitle('Device selection')
        device_gb_layout = qwt.QVBoxLayout(device_gb)
        device_row = HWidget()
        device_gb_layout.addWidget(device_row)
        device_row.layout().setContentsMargins(0, 0, 0, 0)
        self.device_bg = qwt.QButtonGroup(device_row)
        for device in APP_CONFIG.app.core.devices:
            btn = qwt.QRadioButton(device.value, device_gb)
            if device == DEVICE.CPU:
                btn.setChecked(True)
            btn.toggled.connect(self._device_changed)
            device_row.layout().addWidget(btn)
            self.device_bg.addButton(btn)

        left_part.layout().addItem(VerticalSpacer)

        start_btn = qwt.QPushButton(text='start')
        self._widgets_to_disable_on_inference.append(start_btn)
        start_btn.clicked.connect(self._start_inference)
        left_part.layout().addWidget(start_btn)

        right_part = VWidget()
        layout.addWidget(right_part)

        self.preview = Preview(['input image', 'model output', 'merged'], 1)
        right_part.layout().addWidget(self.preview)

    @qtc.pyqtSlot()
    def _on_inference_finished(self) -> None:
        for wgt in self._widgets_to_disable_on_inference:
            self.enable_widget(wgt, True)

    @qtc.pyqtSlot()
    def _on_inference_start(self) -> None:
        for wgt in self._widgets_to_disable_on_inference:
            self.enable_widget(wgt, False)

    @qtc.pyqtSlot()
    def _start_inference(self) -> None:
        if self._image_path is None:
            return
        image = Image.load(self._image_path)
        self._inference_worker.image_sig.emit(image)

    @qtc.pyqtSlot(torch.Tensor, torch.Tensor, torch.Tensor)
    def _inference_result(
        self,
        input_image: torch.Tensor,
        predicted_image: torch.Tensor,
        clone: torch.Tensor,
    ) -> None:
        self.preview.refresh_data_sig.emit(
            [
                [input_image],
                [predicted_image],
                [clone],
            ]
        )

    @qtc.pyqtSlot()
    def _face_detection_algorithm_changed(self) -> None:
        sender = self.sender()
        if sender.isChecked():
            self._inference_worker.algorithm_sig.emit(
                FACE_DETECTION_ALGORITHM[sender.text().upper()]
            )

    @qtc.pyqtSlot()
    def _device_changed(self) -> None:
        sender = self.sender()
        if sender.isChecked():
            self._inference_worker.device_sig.emit(
                DEVICE[sender.text().upper()]
            )

    def _start_inference_thread(self) -> None:
        self._inference_worker = InferenceWorker()
        self._inference_thread = qtc.QThread()
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.start()

    def _load_model(self) -> None:
        # model_path, _ = qwt.QFileDialog.getOpenFileName(
        #     self, 'Select model file', self._last_model_folder
        #     if self._last_model_folder is not None else './models',
        #     'Models (*.p)',
        # )
        model_path = r'C:\Users\tonkec\Documents\deepfake\models\best_model_300.pt'
        if not model_path:
            logger.warning('No model was selected.')
            return

        logger.debug(f'Selected model path: {model_path}.')
        self._last_model_folder = str(Path(model_path).parent.absolute())
        self._inference_worker.model_sig.emit(model_path)

    def _load_image(self) -> None:
        image_path, _ = qwt.QFileDialog.getOpenFileName(
            self, 'Select image file', self._last_image_folder
            if self._last_image_folder is not None else './',
            'Image files (*.png *.jpg)',
        )
        if not image_path:
            logger.warning('No image was selected.')
            return

        logger.debug(f'Selected image: {image_path}')
        self._last_image_folder = str(Path(image_path).parent.absolute())
        self._image_path = image_path
