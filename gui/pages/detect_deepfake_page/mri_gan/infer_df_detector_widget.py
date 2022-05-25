import logging
import multiprocessing
from typing import Any, Dict, Optional, Tuple

import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from common_structures import Job
from core.worker import Worker, InferDFDetectorWorker
from enums import (
    JOB_DATA_KEY,
    JOB_TYPE,
    MRI_GAN_DATASET,
    NUMBER_TYPE,
    OUTPUT_KEYS,
    SIGNAL_OWNER,
    WIDGET_TYPE,
)
from gui.pages.detect_deepfake_page.mri_gan.common import (
    DragAndDrop,
    Parameter,
)
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    ApplyIcon,
    Button,
    CancelIconButton,
    DeviceWidget,
    GroupBox,
    HWidget,
    HorizontalSpacer,
    NoMarginLayout,
    VerticalSpacer,
)
from gui.widgets.dialog import InfoDialog
from utils import parse_number, prepare_path
from variables import DATA_ROOT

logger = logging.getLogger(__name__)


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
            WIDGET_TYPE.RADIO_BUTTON,
        )
        df_detection_model_gb.layout().addWidget(self.df_detection_model)

        self.dad = DragAndDrop('Drop or select video')
        layout.addWidget(self.dad)
        self.dad.installEventFilter(self)
        policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Policy.Expanding,
            qwt.QSizePolicy.Policy.Expanding,
        )
        self.dad.setSizePolicy(policy)

        # select_model_row = HWidget()
        # layout.addWidget(select_model_row)
        # select_model_row.layout().setContentsMargins(0, 0, 0, 0)
        # self._select_model_btn = Button('select')
        # select_model_row.layout().addWidget(self._select_model_btn)
        # self._select_model_btn.clicked.connect(self._select_model)
        # self._model_loaded_ibtn = CancelIconButton()
        # select_model_row.layout().addWidget(self._model_loaded_ibtn)
        # self._model_loaded_lbl = qwt.QLabel(text='model NOT loaded')
        # select_model_row.layout().addWidget(self._model_loaded_lbl)
        # select_model_row.layout().addItem(HorizontalSpacer())

        self._start_inference_btn = Button('start inference')
        layout.addWidget(self._start_inference_btn)
        self._start_inference_btn.clicked.connect(self._start_inference)

        self.setMaximumWidth(400)

        # layout.addItem(VerticalSpacer())

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
        self.new_job_sig.connect(
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
    def _on_infer_df_detector_worker_output(self, res: Dict[OUTPUT_KEYS, Any]) -> None:
        fake_prob = res.get(OUTPUT_KEYS.FAKE_PROB, -1)
        real_prob = res.get(OUTPUT_KEYS.REAL_PROB, -1)
        pred = res.get(OUTPUT_KEYS.PREDICTION, -1)
        msg = f'Fake prob: {fake_prob}, real prob: {real_prob}, pred: {pred}'
        diag = InfoDialog('Inference result', msg)
        diag.exec()

    def _select_image(self) -> None:
        # path = qwt.QFileDialog.getOpenFileName(self, 'Select an image')
        # if path != ('', ''):
        #     path = path[0]
        # else:
        #     return
        # self.dad.image_path_sig.emit(path)
        path = r'C:\Users\tonkec\Desktop\aaoqanfmgd.mp4'
        job = Job(
            {
                JOB_DATA_KEY.FILE_PATH: path,
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
