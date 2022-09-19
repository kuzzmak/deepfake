from collections import namedtuple
from pathlib import Path
import os
import sys
import subprocess
from typing import Dict, Optional, Union

import cv2 as cv
from PIL import Image, ImageQt
import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import Button, NoMarginLayout


ModelRun = namedtuple('ModelRun', ['model_name', 'run_id'])


class TrainingPreview(BaseWidget):

    selected_run_sig = qtc.pyqtSignal(ModelRun)

    def __init__(
        self,
        samples_dir: Union[str, Path],
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._samples_dir = Path(samples_dir)

        self.selected_run_sig.connect(self._selected_run_changed)

        self._current_img_path = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        self._img_lbl = qwt.QLabel(text='halo')
        layout.addWidget(self._img_lbl)

        self._view_in_photo_viewer_btn = Button('View in photo viewer')
        layout.addWidget(self._view_in_photo_viewer_btn)
        self._view_in_photo_viewer_btn.clicked.connect(
            self._view_in_photo_viewer
        )

    @qtc.pyqtSlot(ModelRun)
    def _selected_run_changed(self, run: ModelRun) -> None:
        samples_path = self._samples_dir / run.model_name.value / run.run_id
        samples = list(samples_path.glob('*.jpg'))
        # sort by modification date, first file is the newest
        samples = sorted(samples, key=os.path.getmtime, reverse=True)
        if len(samples):
            img = cv.imread(str(samples[0]))
            img = cv.resize(img, (500, 500), cv.INTER_CUBIC)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img, mode='RGB')
            qt_img = ImageQt.ImageQt(img)
            self._img_lbl.setPixmap(qtg.QPixmap.fromImage(qt_img))
            self._current_img_path = str(samples[0])

    @qtc.pyqtSlot()
    def _view_in_photo_viewer(self) -> None:
        if self._current_img_path is None:
            return
        imageViewerFromCommandLine = {
            'linux': 'xdg-open',
            'win32': 'explorer',
            'darwin': 'open',
        }[sys.platform]
        subprocess.run([imageViewerFromCommandLine, self._current_img_path])
