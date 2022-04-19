import logging
import multiprocessing
import os
from typing import Optional

import PyQt6.QtCore as qtc

from core.df_detection.mri_gan.data_utils.face_detection import \
    crop_faces_from_video
from core.worker import MRIGANWorker
from core.worker.worker_with_pool import WorkerWithPool
from enums import DATA_TYPE, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages


class CropFacesWorker(MRIGANWorker, WorkerWithPool):
    """Worker for cropping faces from images.

    Args:
        num_instances (int, optional): how many instances of this worker
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
            'Started cropping faces process on DFDC ' +
            f'{self._data_type.value} dataset.'
        )
        data_paths = self._get_data_paths()
        self.logger.info(f'Found {len(data_paths)} videos in directory.')
        landmarks_path = self._get_dfdc_landmarks_data_paths()
        self.logger.info(
            f'Using landmarks file from {landmarks_path} directory.'
        )
        crops_path = self._get_dfdc_crops_data_path()
        self.logger.info(
            f'Cropped faces will be saved in {crops_path} directory.'
        )
        os.makedirs(crops_path, exist_ok=True)

        with multiprocessing.Pool(self._num_instances) as pool:
            jobs = []
            results = []
            for dp in data_paths:
                jobs.append(
                    pool.apply_async(
                        crop_faces_from_video,
                        (
                            dp,
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

                self.report_progress(
                    SIGNAL_OWNER.CROPPING_FACES_WORKER,
                    JOB_TYPE.CROPING_FACES,
                    idx,
                    len(jobs),
                )

        self.logger.info('Face cropping finished.')
