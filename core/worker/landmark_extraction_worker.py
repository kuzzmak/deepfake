import logging
import multiprocessing
from multiprocessing.pool import AsyncResult
import os
from typing import List, Optional

import PyQt6.QtCore as qtc

from core.df_detection.mri_gan.data_utils.face_detection import \
    extract_landmarks_from_video
from core.worker import MRIGANWorker, WorkerWithPool
from enums import DATA_TYPE, JOB_NAME, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages


class LandmarkExtractionWorker(MRIGANWorker, WorkerWithPool):
    """Worker used to extact face landmarks from the images with faces.

    Args:
        data_type (DATA_TYPE): for what kind of data is dataset being
            generated
        num_processes (int, optional): how many instances of the worker
            will be spawned. Defaults to 2.
        message_worker_sig (Optional[qtc.pyqtSignal], optional): signal to
            the message worker. Defaults to None.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        data_type: DATA_TYPE,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        MRIGANWorker.__init__(self, data_type)
        WorkerWithPool.__init__(self, num_instances, message_worker_sig)

    def run_job(self) -> None:
        self.logger.info(
            'Started landmark extraction process on DFDC ' +
            f'{self._data_type.value} dataset.'
        )
        data_paths = self._get_data_paths()
        self.logger.info(f'Found {len(data_paths)} videos in directory.')
        out_dir = self._get_dfdc_landmarks_data_path()
        self.logger.info(f'Landmarks metadata will be saved in {out_dir}.')

        self.logger.debug(
            'Launching landmark extraction with ' +
            f'{self._num_instances} processes.'
        )

        with multiprocessing.Pool(self._num_instances) as pool:
            jobs: List[AsyncResult] = []
            results = []
            for dp in data_paths:
                jobs.append(
                    pool.apply_async(
                        extract_landmarks_from_video,
                        (dp, out_dir,),
                    )
                )

            conf_wgt_msg = Messages.CONFIGURE_WIDGET(
                SIGNAL_OWNER.LANDMARK_EXTRACTION_WORKER,
                WIDGET.JOB_PROGRESS,
                'setMaximum',
                [len(jobs)],
                JOB_NAME.LANDMARK_EXTRACTION,
            )
            self.send_message(conf_wgt_msg)

            self.running.emit()

            for idx, job in enumerate(jobs):
                # possible solution where no queue is used
                # qwt.QApplication.processEvents(
                #     qtc.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                # )
                if self.should_exit():
                    self.handle_exit(pool)
                    return

                results.append(job.get())

                self.report_progress(
                    SIGNAL_OWNER.LANDMARK_EXTRACTION_WORKER,
                    JOB_TYPE.LANDMARK_EXTRACTION,
                    idx,
                    len(jobs),
                )

        self.logger.info('Landmark extraction finished.')
