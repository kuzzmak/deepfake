from asyncio.log import logger
import logging
import multiprocessing
from multiprocessing import Queue
from multiprocessing.queues import Empty
from multiprocessing.pool import AsyncResult
import os
from typing import Dict, List, Optional

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt
from tqdm import tqdm

from core.df_detection.mri_gan.data_utils.face_detection import \
    extract_landmarks_from_video
from core.df_detection.mri_gan.data_utils.utils import \
    get_dfdc_training_video_filepaths
from core.df_detection.mri_gan.utils import ConfigParser
from enums import (
    BODY_KEY,
    CONNECTION,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)
from gui.pages.detect_deepfake_page.model_widget import ModelWidget
from gui.widgets.common import (
    Button,
    GroupBox,
    HWidget,
    PlayIcon,
    StopIcon,
    VerticalSpacer,
)
from message.message import Body, Message, Messages


class LandmarkExtractionWorker(qtc.QObject):

    started = qtc.pyqtSignal()
    running = qtc.pyqtSignal()
    finished = qtc.pyqtSignal()

    logger = logging.getLogger('LandmarkExtraction')

    def __init__(
        self,
        num_processes: int = 2,
        conn_q: Optional[Queue] = None,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__()

        self._num_processes = num_processes
        self._conn_q = conn_q
        self._message_worker_sig = message_worker_sig

    def _should_exit(self) -> bool:
        if self._conn_q is None:
            return False
        try:
            _ = self._conn_q.get_nowait()
            return True
        except Empty:
            return False

    @staticmethod
    def _close_pool(pool: multiprocessing.pool.Pool) -> None:
        logger.debug('Closing process pool.')
        pool.close()
        pool.terminate()
        pool.join()
        logger.debug('Process pool closed.')

    @qtc.pyqtSlot()
    def run(self) -> None:
        self.started.emit()
        self.logger.info(
            'Started landmark extraction process for DFDC dataset.'
        )
        data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
        logger.info(
            f'Starting landmark extraction in directory: {data_path_root}.'
        )
        file_paths = get_dfdc_training_video_filepaths(data_path_root)
        logger.info(f'Found {len(file_paths)} videos in directory.')
        out_dir = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
        logger.info(f'Landmarks metadata will be written in {out_dir}.')
        os.makedirs(out_dir, exist_ok=True)

        logger.debug(
            f'Launching landmark extraction with {self._num_processes}.'
        )

        with multiprocessing.Pool(self._num_processes) as pool:
            jobs: List[AsyncResult] = []
            results = []
            for fp in file_paths:
                jobs.append(
                    pool.apply_async(
                        extract_landmarks_from_video,
                        (fp, out_dir,),
                    )
                )
            if self._message_worker_sig is not None:
                conf_wgt_msg = Messages.CONFIGURE_WIDGET(
                    SIGNAL_OWNER.MRI_GAN_WIDGET,
                    WIDGET.JOB_PROGRESS,
                    'setMaximum',
                    [len(jobs)],
                )
                self._message_worker_sig.emit(conf_wgt_msg)

            self.running.emit()

            for idx, job in enumerate(tqdm(jobs, desc="Extracting landmarks")):
                # possible solution where no queue is used
                # qwt.QApplication.processEvents(
                #     qtc.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                # )
                if self._should_exit():
                    logger.debug('Received stop signal, exiting now.')
                    LandmarkExtractionWorker._close_pool(pool)
                    self.finished.emit()
                    return

                results.append(job.get())

                if self._message_worker_sig is not None:
                    job_prog_msg = Message(
                        MESSAGE_TYPE.ANSWER,
                        MESSAGE_STATUS.OK,
                        SIGNAL_OWNER.IMAGE_VIEWER,
                        SIGNAL_OWNER.JOB_PROGRESS,
                        Body(
                            JOB_TYPE.LANDMARK_EXTRACTION,
                            {
                                BODY_KEY.PART: idx,
                                BODY_KEY.TOTAL: len(jobs),
                                BODY_KEY.JOB_NAME: 'landmark extraction'
                            },
                            idx == len(jobs) - 1,
                        )
                    )
                    self._message_worker_sig.emit(job_prog_msg)

        LandmarkExtractionWorker._close_pool(pool)
        logger.info('Landmark extraction finished.')
        self.finished.emit()


class MriGanWidget(ModelWidget):

    stop_landmark_extraction_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals)

        self._init_ui()

        self._threads = []
        self._lmrks_extraction_in_progress = False

    def _init_ui(self) -> None:

        lmrks_extraction_gb = GroupBox('Landmark extraction')
        self.data_tab.layout().addWidget(lmrks_extraction_gb)

        num_of_proc_row = HWidget()
        lmrks_extraction_gb.layout().addWidget(num_of_proc_row)
        num_of_proc_row.setMaximumWidth(200)
        num_of_proc_row.layout().setContentsMargins(0, 0, 0, 0)
        num_of_proc_row.layout().addWidget(qwt.QLabel(
            text='number or processes'
        ))
        self.num_proc_input = qwt.QLineEdit()
        num_of_proc_row.layout().addWidget(self.num_proc_input)

        self.extract_landmarks_btn = Button('start extraction')
        lmrks_extraction_gb.layout().addWidget(self.extract_landmarks_btn)
        self.extract_landmarks_btn.setIcon(PlayIcon())
        self.extract_landmarks_btn.clicked.connect(self._extract_landmarks)

        self.data_tab.layout().addItem(VerticalSpacer)

    @qtc.pyqtSlot()
    def _extract_landmarks(self) -> None:
        """Initiates landmark extraction process or stopps it.
        """
        if not self._lmrks_extraction_in_progress:
            thread = qtc.QThread()
            conn_q = Queue()
            worker = LandmarkExtractionWorker(
                conn_q=conn_q,
                message_worker_sig=self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
            )
            self.stop_landmark_extraction_sig.connect(
                lambda: conn_q.put(CONNECTION.STOP)
            )
            worker.moveToThread(thread)
            self._threads.append((thread, worker))
            thread.started.connect(worker.run)
            worker.started.connect(
                self._on_landmark_extraction_worker_started
            )
            worker.running.connect(
                self._on_landmark_extraction_worker_running
            )
            worker.finished.connect(
                self._on_landmark_extraction_worker_finished
            )
            thread.start()
        else:
            self._stop_landmark_extraction()

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_finished(self) -> None:
        """Waits for landmark extraction thread to finish and exit gracefully.
        """
        for thread, _ in self._threads:
            thread.quit()
            thread.wait()
        self._threads = []
        self.enable_widget(self.extract_landmarks_btn, True)
        self.extract_landmarks_btn.setIcon(PlayIcon())
        self.extract_landmarks_btn.setText('start extraction')
        self._lmrks_extraction_in_progress = False

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_started(self) -> None:
        """pyqtSlot which disables button for starting or stopping landmark
        extraction until worker is set up.
        """
        self.enable_widget(self.extract_landmarks_btn, False)
        self._lmrks_extraction_in_progress = True

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_running(self) -> None:
        """pyqtSlot which enables button for starting or stopping landmark
        extraction and changes icon of the button to the stop icon.
        """
        self.enable_widget(self.extract_landmarks_btn, True)
        self.extract_landmarks_btn.setIcon(StopIcon())
        self.extract_landmarks_btn.setText('stop extraction')

    def _stop_landmark_extraction(self) -> None:
        self.enable_widget(self.extract_landmarks_btn, False)
        self.stop_landmark_extraction_sig.emit()
