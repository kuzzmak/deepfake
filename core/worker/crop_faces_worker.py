import logging
import multiprocessing
import os
from typing import Optional

import PyQt6.QtCore as qtc
from tqdm import tqdm

from core.df_detection.mri_gan.data_utils.face_detection import \
    crop_faces_from_video
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


class CroppingFacesWorker(WorkerWithPool):

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(num_instances, message_worker_sig)

    def run_job(self) -> None:
        self.logger.info('Started cropping faces for DFDC dataset.')
        data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
        self.logger.info(f'Starting cropping in {data_path_root} directory.')
        file_paths = get_dfdc_training_video_filepaths(data_path_root)
        self.logger.info(f'Found {len(file_paths)} videos in directory.')
        landmarks_path = ConfigParser \
            .getInstance() \
            .get_dfdc_landmarks_train_path()
        self.logger.info(
            f'Using landmarks file from {landmarks_path} directory.'
        )
        crops_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
        self.logger.info(
            f'Cropped faces will be saved in {crops_path} directory.'
        )
        os.makedirs(crops_path, exist_ok=True)

        with multiprocessing.Pool(self._num_instances) as pool:
            jobs = []
            results = []
            for input_filepath in file_paths:
                jobs.append(
                    pool.apply_async(
                        crop_faces_from_video,
                        (
                            input_filepath,
                            landmarks_path,
                            crops_path,
                        ),
                    )
                )

            if self._message_worker_sig is not None:
                conf_wgt_msg = Messages.CONFIGURE_WIDGET(
                    SIGNAL_OWNER.CROPPING_FACES_WORKER,
                    WIDGET.JOB_PROGRESS,
                    'setMaximum',
                    [len(jobs)],
                )
                self.send_message(conf_wgt_msg)

            self.running.emit()

            for idx, job in enumerate(jobs):
                if self.should_exit():
                    self.handle_exit(pool)
                    return

                results.append(job.get())

                if self._message_worker_sig is not None:
                    job_prog_msg = Message(
                        MESSAGE_TYPE.ANSWER,
                        MESSAGE_STATUS.OK,
                        SIGNAL_OWNER.CROPPING_FACES_WORKER,
                        SIGNAL_OWNER.JOB_PROGRESS,
                        Body(
                            JOB_TYPE.CROPING_FACES,
                            {
                                BODY_KEY.PART: idx,
                                BODY_KEY.TOTAL: len(jobs),
                                BODY_KEY.JOB_NAME: 'cropping faces'
                            },
                            idx == len(jobs) - 1,
                        )
                    )
                    self.send_message(job_prog_msg)

        self.logger.info('Face cropping finished.')
