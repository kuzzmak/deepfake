from collections import namedtuple
import logging
from pathlib import Path
import os
import sys
import subprocess
from typing import Dict, Optional, Union

import cv2 as cv
import numpy as np
from PIL import Image, ImageQt
import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import Button, HWidget, HorizontalSpacer, MinimalSizePolicy, NoMarginLayout, VerticalSpacer
from utils import resize_image_retain_aspect_ratio
from variables import APP_LOGGER


ModelRun = namedtuple('ModelRun', ['model_name', 'run_id'])
OFFSET = 200
logger = logging.getLogger(APP_LOGGER)

class TrainingPreview(BaseWidget):

    selected_run_sig = qtc.pyqtSignal(ModelRun)
    new_sample_sig = qtc.pyqtSignal(Path)

    def __init__(
        self,
        samples_dir: Union[str, Path],
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._samples_dir = Path(samples_dir)

        self.selected_run_sig.connect(self._selected_run_changed)
        self.new_sample_sig.connect(self._new_sample)

        self._current_img_path = None
        self._current_img_idx = 0
        self._preview_image = None
        self._empty_pixmap = qtg.QPixmap()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        run_samples_cb_row = HWidget()
        layout.addWidget(run_samples_cb_row)
        lbl = qwt.QLabel(text='current sample')
        lbl.setSizePolicy(MinimalSizePolicy())
        run_samples_cb_row.layout().addWidget(lbl)
        self._run_samples_cb = qwt.QComboBox()
        run_samples_cb_row.layout().addWidget(self._run_samples_cb)
        self._run_samples_cb.currentIndexChanged.connect(
            self._selected_run_sample_changed
        )

        layout.addItem(VerticalSpacer())

        self._img_lbl = qwt.QLabel()
        layout.addWidget(
            self._img_lbl,
            alignment=qtc.Qt.AlignmentFlag.AlignCenter,
        )

        button_row = HWidget()
        layout.addWidget(button_row)
        button_row.layout().addItem(HorizontalSpacer())
        self._view_in_photo_viewer_btn = Button('view in photo viewer')
        button_row.layout().addWidget(self._view_in_photo_viewer_btn)
        self._view_in_photo_viewer_btn.clicked.connect(
            self._view_in_photo_viewer
        )
        previous_sample_btn = Button('previous sample')
        button_row.layout().addWidget(previous_sample_btn)
        previous_sample_btn.clicked.connect(self._previous_preview_image)
        next_sample_btn = Button('next sample')
        button_row.layout().addWidget(next_sample_btn)
        next_sample_btn.clicked.connect(self._next_preview_image)
        button_row.layout().addItem(HorizontalSpacer())

    @qtc.pyqtSlot(int)
    def _selected_run_sample_changed(self, index: int) -> None:
        self._current_img_idx = index
        self._refresh_preview_image()

    def resizeEvent(self, event: qtg.QResizeEvent) -> None:
        self._set_preview_image(
            self._preview_image, 
            qtc.QSize(
                self.size().width() - OFFSET,
                self.size().height() - OFFSET,
            ),
        )
        return super().resizeEvent(event)

    def _set_preview_image(
        self,
        image: np.ndarray,
        size: Optional[qtc.QSize] = None,
    ) -> None:
        if self._preview_image is None or image is None:
            self._img_lbl.setPixmap(self._empty_pixmap)
            return
        max_size = max([size.height(), size.width()]) \
            if size is not None else 400
        image = resize_image_retain_aspect_ratio(np.copy(image), max_size)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image, mode='RGB')
        self._qt_img = ImageQt.ImageQt(image)
        self._img_lbl.setPixmap(qtg.QPixmap.fromImage(self._qt_img))
        logger.debug(f'Previewing image on path: {self._current_img_path}.')

    @qtc.pyqtSlot(Path)
    def _new_sample(self, sample_path: Path) -> None:
        self._samples.insert(0, sample_path)
        self._current_img_idx = 0
        self._refresh_preview_image()
        self._refresh_samples_dropdown()

    def _refresh_preview_image(self) -> None:
        # TODO refresh correct run image when resuming run
        if not len(self._samples):
            self._set_preview_image(None)
            return
        path = str(self._samples[self._current_img_idx])
        if path == self._current_img_path:
            return
        self._current_img_path = path
        self._preview_image = cv.imread(self._current_img_path)
        self._set_preview_image(
            self._preview_image,
            qtc.QSize(
                self.size().width() - OFFSET,
                self.size().height() - OFFSET,
            ),
        )

    @qtc.pyqtSlot()
    def _next_preview_image(self) -> None:
        if self._current_img_idx - 1 < 0:
            return
        self._current_img_idx -= 1
        self._refresh_preview_image()
        self._run_samples_cb.setCurrentIndex(self._current_img_idx)

    @qtc.pyqtSlot()
    def _previous_preview_image(self) -> None:
        if self._current_img_idx + 1 >= len(self._samples):
            return
        self._current_img_idx += 1
        self._refresh_preview_image()
        self._run_samples_cb.setCurrentIndex(self._current_img_idx)

    def _refresh_samples_dropdown(self) -> None:
        self._run_samples_cb.clear()
        self._run_samples_cb.addItems(
            list(map(lambda p: p.name, self._samples))
        )

    @qtc.pyqtSlot(ModelRun)
    def _selected_run_changed(self, run: ModelRun) -> None:
        if run.run_id is None:
            return
        samples_path = self._samples_dir / run.model_name.value / run.run_id
        samples = list(samples_path.glob('*.jpg'))
        # sort by modification date, first file is the newest
        self._samples = sorted(samples, key=os.path.getmtime, reverse=True)
        self._current_img_idx = 0
        self._refresh_preview_image()
        self._refresh_samples_dropdown()

    @qtc.pyqtSlot()
    def _view_in_photo_viewer(self) -> None:
        if self._current_img_path is None:
            return
        im_viewer = {
            'linux': 'xdg-open',
            'win32': 'explorer',
            'darwin': 'open',
        }[sys.platform]
        subprocess.run([im_viewer, self._current_img_path])
