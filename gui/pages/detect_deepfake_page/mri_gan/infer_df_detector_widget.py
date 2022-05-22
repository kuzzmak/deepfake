from typing import Dict, Optional, Tuple

import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from common_structures import Job
from core.worker import Worker, InferDFDetectorWorker
from enums import JOB_DATA_KEY, JOB_TYPE, SIGNAL_OWNER
from gui.pages.detect_deepfake_page.mri_gan.common import DragAndDrop
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    ApplyIcon,
    Button,
    CancelIconButton,
    DeviceWidget,
    HWidget,
    HorizontalSpacer,
    NoMarginLayout,
    VerticalSpacer,
)
from variables import DATA_ROOT


class InferDFDetectorWidget(BaseWidget):

    new_job_sig = qtc.pyqtSignal(Job)

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        self.devices = DeviceWidget()
        layout.addWidget(self.devices)

        self.dad = DragAndDrop()
        layout.addWidget(self.dad)
        self.dad.installEventFilter(self)

        select_model_row = HWidget()
        layout.addWidget(select_model_row)
        select_model_row.layout().setContentsMargins(0, 0, 0, 0)
        self._select_model_btn = Button('select')
        select_model_row.layout().addWidget(self._select_model_btn)
        self._select_model_btn.clicked.connect(self._select_model)
        self._model_loaded_ibtn = CancelIconButton()
        select_model_row.layout().addWidget(self._model_loaded_ibtn)
        self.model_loaded_lbl = qwt.QLabel(text='model NOT loaded')
        select_model_row.layout().addWidget(self.model_loaded_lbl)
        select_model_row.layout().addItem(HorizontalSpacer())

        self._start_inference_btn = Button('start inference')
        layout.addWidget(self._start_inference_btn)
        self._start_inference_btn.clicked.connect(self._start_inference)

        self.setMaximumWidth(400)

        layout.addItem(VerticalSpacer())

    @qtc.pyqtSlot()
    def _start_inference(self) -> None:
        thread = qtc.QThread()
        worker = InferDFDetectorWorker(
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
        )
        self.new_job_sig.connect(
            lambda job: worker.job_q.put(job)
        )
        worker.moveToThread(thread)
        self._threads[JOB_TYPE.INFER_DF_DETECTOR] = (thread, worker)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_infer_df_detector_worker_finished)
        thread.start()

    @qtc.pyqtSlot()
    def _on_infer_df_detector_worker_finished(self) -> None:
        val = self._threads.get(JOB_TYPE.INFER_DF_DETECTOR, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.INFER_DF_DETECTOR, None)

    @qtc.pyqtSlot()
    def _select_model(self) -> None:
        path = qwt.QFileDialog.getOpenFileName(
            self,
            'Select model',
            str(DATA_ROOT),
            'p(*.p)',
        )
        if path != ('', ''):
            path = path[0]
        else:
            return
        self._model_loaded_ibtn.setIcon(ApplyIcon())
        self._select_model_btn.setText('model loaded')

    def _select_image(self) -> None:
        path = qwt.QFileDialog.getOpenFileName(self, 'Select an image')
        if path != ('', ''):
            path = path[0]
        else:
            return
        self.dad.image_path_sig.emit(path)
        job = Job(
            {
                JOB_DATA_KEY.IMAGE_PATH: path,
            }
        )
        self.new_job_sig.emit(job)

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent):
        if event.type() == qtc.QEvent.Type.Enter:
            self.setCursor(qtg.QCursor(qtc.Qt.CursorShape.PointingHandCursor))
        elif event.type() == qtc.QEvent.Type.Leave:
            self.setCursor(qtg.QCursor(qtc.Qt.CursorShape.ArrowCursor))
        elif event.type() == qtc.QEvent.Type.MouseButtonPress:
            self._select_image()
        return super().eventFilter(source, event)
