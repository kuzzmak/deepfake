import logging
import multiprocessing
from multiprocessing.pool import AsyncResult
import os
from typing import List, Optional

import PyQt6.QtCore as qtc

from core.df_detection.mri_gan.data_utils.face_detection import \
    extract_landmarks_from_video
from core.df_detection.mri_gan.data_utils.utils import \
    get_dfdc_training_video_filepaths
from core.df_detection.mri_gan.utils import ConfigParser
from core.worker.worker_with_pool import WorkerWithPool
from enums import (
    BODY_KEY,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)
from message.message import Body, Message, Messages


class LandmarkExtractionWorker(WorkerWithPool):
    """Worker used to extact face landmarks from the images with faces.

    Args:
        num_processes (int, optional): how many instances of the worker
            will be spawned. Defaults to 2.
        message_worker_sig (Optional[qtc.pyqtSignal], optional): signal to
            the message worker. Defaults to None.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(num_instances, message_worker_sig)

    def run_job(self) -> None:
        self.logger.info(
            'Started landmark extraction process for DFDC dataset.'
        )
        data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
        self.logger.info(
            f'Starting landmark extraction in directory: {data_path_root}.'
        )
        file_paths = get_dfdc_training_video_filepaths(data_path_root)
        self.logger.info(f'Found {len(file_paths)} videos in directory.')
        out_dir = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
        self.logger.info(f'Landmarks metadata will be saved in {out_dir}.')
        os.makedirs(out_dir, exist_ok=True)

        self.logger.debug(
            f'Launching landmark extraction with ' +
            f'{self._num_instances} processes.'
        )

        with multiprocessing.Pool(self._num_instances) as pool:
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
                    SIGNAL_OWNER.LANDMARK_EXTRACTION_WORKER,
                    WIDGET.JOB_PROGRESS,
                    'setMaximum',
                    [len(jobs)],
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

                if self._message_worker_sig is not None:
                    job_prog_msg = Message(
                        MESSAGE_TYPE.ANSWER,
                        MESSAGE_STATUS.OK,
                        SIGNAL_OWNER.LANDMARK_EXTRACTION_WORKER,
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
                    self.send_message(job_prog_msg)

        self.logger.info('Landmark extraction finished.')
