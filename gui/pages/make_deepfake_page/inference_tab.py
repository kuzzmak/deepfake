import logging
from pathlib import Path
from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
from config import APP_CONFIG

from core.image.image import Image
from gui.workers.inference_worker import InferenceWorker
from enums import DEVICE, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import HWidget, VWidget, VerticalSpacer

logger = logging.getLogger(__name__)


class InferenceTab(BaseWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
    ):
        super().__init__(signals)
        self._start_inference_thread()
        self._init_ui()
        self._threads = []
        self._last_model_folder = None
        self._last_image_folder = None

    def _init_ui(self):
        layout = qwt.QHBoxLayout()
        self.setLayout(layout)
        left_part = VWidget()
        layout.addWidget(left_part)
        left_part.setMaximumWidth(300)

        model_gb = qwt.QGroupBox()
        left_part.layout().addWidget(model_gb)
        model_gb.setTitle('Model selection')
        model_gb_layout = qwt.QHBoxLayout(model_gb)
        model_select_btn = qwt.QPushButton(text='select')
        model_gb_layout.addWidget(model_select_btn)
        model_select_btn.clicked.connect(self._load_model)

        image_gb = qwt.QGroupBox()
        left_part.layout().addWidget(image_gb)
        image_gb.setTitle('Image selection')
        image_gb_layout = qwt.QHBoxLayout(image_gb)
        image_select_btn = qwt.QPushButton(text='select')
        image_gb_layout.addWidget(image_select_btn)
        image_select_btn.clicked.connect(self._load_image)

        device_gb = qwt.QGroupBox()
        left_part.layout().addWidget(device_gb)
        device_gb.setTitle('Device selection')
        device_gb_layout = qwt.QVBoxLayout(device_gb)
        device_row = HWidget()
        device_gb_layout.addWidget(device_row)
        device_row.layout().setContentsMargins(0, 0, 0, 0)
        self.device_bg = qwt.QButtonGroup(device_row)
        for device in APP_CONFIG.app.core.devices:
            btn = qwt.QRadioButton(device.value, device_gb)
            btn.setChecked(True)
            btn.toggled.connect(self._device_changed)
            device_row.layout().addWidget(btn)
            self.device_bg.addButton(btn)

        left_part.layout().addItem(VerticalSpacer)

        right_part = VWidget()
        layout.addWidget(right_part)

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
        model_path = r'C:\Users\kuzmi\Documents\deepfake\models\best_model_0.pt'
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

        image = Image.load(image_path)
        self._inference_worker.image_sig.emit(image)
