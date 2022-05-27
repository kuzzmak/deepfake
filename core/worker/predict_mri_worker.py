import logging
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image
import PyQt6.QtCore as qtc
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image

from configs.mri_gan_config import MRIGANConfig
from core.df_detection.mri_gan.data_utils.utils import filter_dfdc_dirs
from core.df_detection.mri_gan.mri_gan.model import get_MRI_GAN
from core.worker import MRIGANWorker, WorkerWithPool
from enums import DATA_TYPE, DEVICE, JOB_NAME, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages
from utils import batchify

logger = logging.getLogger(__name__)


class PredictMRIWorker(MRIGANWorker, WorkerWithPool):
    """Worker which predicts, using trained MRI gan, MRIs for the whole
    DFDC dataset, i.e. MRIs on cropped faces which will be used to train
    deepfake detector model.

    Parameters
    ----------
    data_type : DATA_TYPE
        which data is currently being predicted
    batch_size : int, optional
        batch size for the MRI gan model, by default 8
    num_instances : int, optional
        number of same processes that will be launched, by default 2
    message_worker_sig : Optional[qtc.pyqtSignal], optional
        signal to the message worker, by default None
    """

    def __init__(
        self,
        data_type: DATA_TYPE,
        batch_size: int = 8,
        num_instances: int = 2,
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        MRIGANWorker.__init__(self, data_type)
        WorkerWithPool.__init__(self, num_instances, message_worker_sig)

        self._batch_size = batch_size
        self._device = device

    @staticmethod
    def _get_video_paths(crops_path: Path) -> List[Path]:
        """Constructs paths for the videos from cropped directory.

        Parameters
        ----------
        crops_path : Path
            directory with cropped faces

        Returns
        -------
        List[Path]
            list of video paths
        """
        dirs = os.listdir(crops_path)
        dirs = filter_dfdc_dirs(dirs)
        dirs = [crops_path / d for d in dirs]
        file_dirs = []
        [
            file_dirs.extend(
                [
                    directory / file for file in os.listdir(directory)
                ]
            )
            for directory in dirs
        ]
        return file_dirs

    @staticmethod
    def _predict_mri_using_MRI_GAN(
        mri_path: Path,
        v_d: Path,
        batch_size: int,
        device: DEVICE = DEVICE.CPU,
        load_model_from_gd=False,
        overwrite=False,
        inference=False,
    ) -> None:
        """Uses trained MRI gan to predict MRI for every cropped face from the
        DFDC dataset in order to create MRI dataset for the deepfake detectio
        model.

        Parameters
        ----------
        mri_path : Path
            directory where predicted MRIs will be saved
        v_d : Path
            path of the video directory with cropped faces
        batch_size : int
            batch size for the prediction
        overwrite : bool, optional
            should already predicted MRIs be overwritten, by default False
        """
        logger.debug(f'Predicting MRI for video {str(v_d)}.')
        video_id = v_d.parts[-1]
        if inference:
            part = ''
        else:
            part = v_d.parts[-2]
        vid_mri_path = mri_path / part / video_id
        if not overwrite and vid_mri_path.is_dir():
            return
        vid_mri_path.mkdir(exist_ok=True)
        frame_paths = [v_d / file for file in os.listdir(v_d)]

        im_size = MRIGANConfig \
            .get_instance() \
            .get_mri_gan_model_params()['imsize']
        transforms_ = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        mri_generator = get_MRI_GAN(
            load_from_gd=load_model_from_gd,
            device=device,
        )

        for frame_names in batchify(frame_paths, batch_size):
            frames = list(
                map(lambda fn: transforms_(Image.open(fn)), frame_names)
            )
            frames = torch.stack(frames)
            frames = frames.to(device.value)
            mri_images = mri_generator(frames)
            for idx in range(mri_images.shape[0]):
                save_path = vid_mri_path / frame_names[idx].parts[-1]
                save_image(mri_images[idx], save_path)

    def run_job(self) -> None:
        logger.info('MRI prediction started.')

        crops_path = self._get_dfdc_crops_data_path()
        video_dir_paths = PredictMRIWorker._get_video_paths(crops_path)
        logger.info(f'Found {len(video_dir_paths)} videos.')

        mri_path = self._get_mrip2p_png_data_path()
        with multiprocessing.Pool(self._num_instances) as pool:
            jobs = []
            for v_d in video_dir_paths:
                jobs.append(
                    pool.apply_async(
                        PredictMRIWorker._predict_mri_using_MRI_GAN,
                        (mri_path, v_d, self._batch_size, self._device),
                    )
                )

            conf_wgt_msg = Messages.CONFIGURE_WIDGET(
                SIGNAL_OWNER.PREDICT_MRI_WORKER,
                WIDGET.JOB_PROGRESS,
                'setMaximum',
                [len(jobs)],
                JOB_NAME.PREDICT_MRI,
            )
            self.send_message(conf_wgt_msg)

            self.running.emit()

            results = []
            for idx, job in enumerate(jobs):
                if self.should_exit():
                    self.handle_exit(pool)
                    return

                results.append(job.get())

                self.report_progress(
                    SIGNAL_OWNER.PREDICT_MRI_WORKER,
                    JOB_TYPE.PREDICT_MRI,
                    idx,
                    len(jobs),
                )

        logger.info('MRI prediction finished.')
