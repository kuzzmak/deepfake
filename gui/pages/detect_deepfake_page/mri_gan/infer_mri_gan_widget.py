from typing import Dict, Optional, Tuple

import PyQt6.QtCore as qtc
from common_structures import Job

from core.worker import InferMRIGANWorker, Worker
from enums import JOB_TYPE, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    Button,
    DeviceWidget,
    GroupBox,
    NoMarginLayout,
    VerticalSpacer,
)


class InferMRIGANWidget(BaseWidget):

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

        self.start_btn = Button('start')
        layout.addWidget(self.start_btn)
        self.start_btn.clicked.connect(self._start_inference)

        layout.addItem(VerticalSpacer())

        self.setFixedWidth(400)

    @qtc.pyqtSlot()
    def _start_inference(self) -> None:
        thread = qtc.QThread()
        worker = InferMRIGANWorker(
            self.devices.device,
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
        )
        self.new_job_sig.connect(
            lambda job: worker.job_q.put(job)
        )
        worker.moveToThread(thread)
        self._threads[JOB_TYPE.INFER_MRI_GAN] = (thread, worker)
        thread.started.connect(worker.run)
        thread.start()

        self.new_job_sig.emit(Job({}))