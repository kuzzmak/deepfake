import logging
import multiprocessing
from typing import Any, Dict, Optional, Tuple

import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
import PyQt6.QtWidgets as qwt

from common_structures import Job
from core.worker import InferDFDetectorWorker, Worker
from enums import (
    JOB_DATA_KEY,
    JOB_TYPE,
    MRI_GAN_DATASET,
    NUMBER_TYPE,
    OUTPUT_KEYS,
    SIGNAL_OWNER,
    WIDGET_TYPE,
)
from gui.pages.detect_deepfake_page.mri_gan.common import DragAndDrop
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    Button,
    DeviceWidget,
    GroupBox,
    HWidget,
    NoMarginLayout,
    Parameter,
)
from utils import parse_number

logger = logging.getLogger(__name__)


class InferDFDetectorWidget(BaseWidget):

    file_change_job_sig = qtc.pyqtSignal(Job)
    df_model_changed_sig = qtc.pyqtSignal(Job)

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

        layout.addWidget(qwt.QLabel(text='deepfake detection model'))

        self.devices = DeviceWidget()
        layout.addWidget(self.devices)

        gb = GroupBox('Parameters')
        layout.addWidget(gb)

        self.fake_threashold_input = Parameter('fake threshold', [0.5])
        gb.layout().addWidget(self.fake_threashold_input)

        self.fake_fraction_input = Parameter('fake fraction', [0.5])
        gb.layout().addWidget(self.fake_fraction_input)

        self.batch_size = Parameter('batch size', [32])
        gb.layout().addWidget(self.batch_size)

        self.num_workers = Parameter(
            'num workers',
            [multiprocessing.cpu_count() // 2],
        )
        gb.layout().addWidget(self.num_workers)

        df_detection_model_gb = GroupBox('Deepfake detection model')
        layout.addWidget(df_detection_model_gb)

        self.df_detection_model = Parameter(
            '',
            [m.value for m in MRI_GAN_DATASET],
            MRI_GAN_DATASET.MRI,
            WIDGET_TYPE.RADIO_BUTTON,
        )
        df_detection_model_gb.layout().addWidget(self.df_detection_model)
        self.df_detection_model._btn_bg.idReleased.connect(
            self._df_model_selection_changed
        )

        self.dad = DragAndDrop('Drop or select video')
        layout.addWidget(self.dad)
        self.dad.installEventFilter(self)
        policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Policy.Expanding,
            qwt.QSizePolicy.Policy.Expanding,
        )
        self.dad.setSizePolicy(policy)
        self.dad.dropped_path_sig.connect(self._emit_new_file_change_job)

        prediction_row = HWidget()
        prediction_row.layout().addWidget(qwt.QLabel(text='fake prob:'))
        self.fake_prob_lbl = qwt.QLabel('-')
        prediction_row.layout().addWidget(self.fake_prob_lbl)
        prediction_row.layout().addWidget(qwt.QLabel(text='prediction:'))
        self.pred_lbl = qwt.QLabel('-')
        prediction_row.layout().addWidget(self.pred_lbl)
        layout.addWidget(prediction_row)

        self._start_inference_btn = Button('start inference')
        layout.addWidget(self._start_inference_btn)
        self._start_inference_btn.clicked.connect(self._start_inference)

        self.setMaximumWidth(400)

    @qtc.pyqtSlot(int)
    def _df_model_selection_changed(self, idx: int) -> None:
        job = Job(
            JOB_TYPE.MODEL_CHANGE,
            {
                JOB_DATA_KEY.MODEL_TYPE:
                MRI_GAN_DATASET[self.df_detection_model.value.upper()]
            }
        )
        self.df_model_changed_sig.emit(job)

    @qtc.pyqtSlot()
    def _start_inference(self) -> None:
        fake_threshold = parse_number(
            self.fake_threashold_input.value,
            NUMBER_TYPE.FLOAT,
        )
        if fake_threshold is None:
            logger.error(
                'Unable to parse fake threshold number, ' +
                'must be float between 0 and 1.'
            )
            return

        fake_fraction = parse_number(
            self.fake_fraction_input.value,
            NUMBER_TYPE.FLOAT,
        )
        if fake_fraction is None:
            logger.error(
                'Unable to parse fake fraction number, ' +
                'must be float between 0 and 1.'
            )
            return

        batch_size = parse_number(self.batch_size.value)
        if batch_size is None:
            logger.error(
                'Unable to parse number for batch ' +
                'size, must be integer.'
            )
            return

        num_workers = parse_number(self.num_workers.value)
        if num_workers is None:
            logger.error(
                'Unable to parse input for num of workers, must ' +
                'be integer and lower than the number of available processors.'
            )
            return

        thread = qtc.QThread()
        worker = InferDFDetectorWorker(
            MRI_GAN_DATASET[self.df_detection_model.value.upper()],
            fake_threshold,
            fake_fraction,
            batch_size,
            num_workers,
            self.devices.device,
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
        )
        self.file_change_job_sig.connect(
            lambda job: worker.job_q.put(job)
        )
        self.df_model_changed_sig.connect(
            lambda job: worker.job_q.put(job)
        )
        worker.moveToThread(thread)
        self._threads[JOB_TYPE.INFER_DF_DETECTOR] = (thread, worker)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_infer_df_detector_worker_finished)
        worker.output.connect(self._on_infer_df_detector_worker_output)
        thread.start()
        self.enable_widget(self._start_inference_btn, False)

    @qtc.pyqtSlot()
    def _on_infer_df_detector_worker_finished(self) -> None:
        val = self._threads.get(JOB_TYPE.INFER_DF_DETECTOR, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.INFER_DF_DETECTOR, None)

    @qtc.pyqtSlot(dict)
    def _on_infer_df_detector_worker_output(
        self,
        res: Dict[OUTPUT_KEYS, Any],
    ) -> None:
        fake_prob = res.get(OUTPUT_KEYS.FAKE_PROB, '-')
        pred = res.get(OUTPUT_KEYS.PREDICTION, '-')
        self.fake_prob_lbl.setText(str(fake_prob))
        self.pred_lbl.setText('FAKE' if pred == 1 else 'REAL')

    def _emit_new_file_change_job(self, path: str) -> None:
        """Sends new job to the infer df worker.

        Parameters
        ----------
        path : str
            path of the file
        """
        job = Job(
            JOB_TYPE.FILE_CHANGE,
            {
                JOB_DATA_KEY.FILE_PATH: path,
            }
        )
        self.file_change_job_sig.emit(job)
        self.fake_prob_lbl.setText('-')
        self.pred_lbl.setText('-')

    def _select_file(self) -> None:
        """Prompts user with the dialog for selecting file for deepfake
        detection.
        """
        path = qwt.QFileDialog.getOpenFileName(self, 'Select an image')
        if path != ('', ''):
            path = path[0]
        else:
            return
        self.dad.set_preview_sig.emit(path)
        self._emit_new_file_change_job(path)

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent):
        if event.type() == qtc.QEvent.Type.Enter:
            self.setCursor(qtg.QCursor(qtc.Qt.CursorShape.PointingHandCursor))
        elif event.type() == qtc.QEvent.Type.Leave:
            self.setCursor(qtg.QCursor(qtc.Qt.CursorShape.ArrowCursor))
        elif event.type() == qtc.QEvent.Type.MouseButtonPress:
            self._select_file()
        return super().eventFilter(source, event)
