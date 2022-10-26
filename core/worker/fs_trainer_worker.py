from pathlib import Path
from queue import Empty
from threading import Thread
from typing import Optional

import PyQt6.QtCore as qtc

from core.trainer.configuration import TrainerConfiguration
from core.trainer.fs_trainer import FSTrainer
from core.worker import Worker
from enums import (
    EVENT_DATA_KEY,
    EVENT_TYPE,
    JOB_NAME,
    JOB_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)
from message.message import Messages


class FSTrainerWorker(Worker):

    new_sample_path_sig = qtc.pyqtSignal(Path)

    def __init__(
        self,
        trainer_conf: TrainerConfiguration,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self.conf = trainer_conf

    def run_job(self) -> None:
        trainer = FSTrainer(self.conf, self.stop_event)

        conf_wgt_msg = Messages.CONFIGURE_WIDGET(
            SIGNAL_OWNER.FS_TRAINER_WORKER,
            WIDGET.JOB_PROGRESS,
            'setMaximum',
            [self.conf.steps],
            JOB_NAME.TRAIN_FS_MODEL,
        )
        self.send_message(conf_wgt_msg)

        def pf():
            while True:
                try:
                    event = trainer.event_q.get()
                    if event.event_type == EVENT_TYPE.PROGRESS:
                        step = event.data[EVENT_DATA_KEY.PROGRESS_VALUE]
                        if step == -1:
                            break
                        self.report_progress(
                            SIGNAL_OWNER.FS_TRAINER_WORKER,
                            JOB_TYPE.TRAIN_FS_DF_MODEL,
                            step,
                            self.conf.steps,
                        )
                    elif event.event_type == EVENT_TYPE.NEW_SAMPLE:
                        sample_path = event.data[EVENT_DATA_KEY.SAMPLE_PATH]
                        self.new_sample_path_sig.emit(sample_path)

                except Empty:
                    ...

        self._pt = Thread(target=pf, daemon=True)
        self._pt.start()

        self.running.emit()
        trainer.start()

    def post_run(self) -> None:
        self._pt.join()
