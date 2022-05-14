import logging
import multiprocessing
from multiprocessing.pool import AsyncResult
import os
from typing import List, Optional

import pandas as pd
import PyQt6.QtCore as qtc
from sklearn.model_selection import train_test_split

from core.df_detection.mri_gan.data_utils.face_mri import \
    gen_face_mri_per_directory
from core.df_detection.mri_gan.data_utils.utils import \
    get_dfdc_training_real_fake_pairs
from core.worker import MRIGANWorker, WorkerWithPool
from enums import DATA_TYPE, JOB_NAME, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages

logger = logging.getLogger(__name__)


class GenerateMRIDatasetWorker(MRIGANWorker, WorkerWithPool):

    def __init__(
        self,
        data_type: DATA_TYPE,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        """Worker for generating MRI dataset.

        Args:
            data_type (DATA_TYPE): for what kind of data is dataset being
                generated
            num_instances (int, optional): number of workers to spawn.
                Defaults to 2.
            message_worker_sig (Optional[qtc.pyqtSignal], optional): signal to
                the message worker. Defaults to None.
        """
        MRIGANWorker.__init__(self, data_type)
        WorkerWithPool.__init__(self, num_instances, message_worker_sig)

    def run_job(self) -> None:
        test_size = 0.2
        dfdc_fract = 0.5
        overwrite = True

        metadata_csv_file = self._get_dfdc_mri_medatata_csv_path()
        logger.info(
            f'Metadata file for {self._data_type.value} ' +
            f'mri dataset location: {str(metadata_csv_file)}.'
        )

        metadata_csv_file.parent.mkdir(exist_ok=True)

        if not overwrite and os.path.isfile(metadata_csv_file):
            return pd.read_csv(metadata_csv_file)

        pairs = get_dfdc_training_real_fake_pairs(self._get_dfdc_data_path())

        mri_basedir = self._get_dfdc_mri_path()
        logger.info(
            f'MRI datasets will be saved in directory: {str(mri_basedir)}.'
        )
        crops_path = self._get_dfdc_crops_data_path()
        logger.debug(
            f'DFDC dataset cropped faces directory: {str(crops_path)}.'
        )

        results = []
        df = pd.DataFrame(columns=['real_image', 'fake_image', 'mri_image'])
        with multiprocessing.Pool(self._num_instances) as pool:
            jobs: List[AsyncResult] = []
            for pid in range(len(pairs)):
                real, fake = pairs[pid]
                reals = crops_path / real
                fakes = crops_path / fake
                jobs.append(
                    pool.apply_async(
                        gen_face_mri_per_directory,
                        (reals, fakes, mri_basedir,),
                    )
                )

            if self._message_worker_sig is not None:
                conf_wgt_msg = Messages.CONFIGURE_WIDGET(
                    SIGNAL_OWNER.GENERATE_MRI_DATASET_WORKER,
                    WIDGET.JOB_PROGRESS,
                    'setMaximum',
                    [len(jobs)],
                    JOB_NAME.GENERATE_MRI_DATASET,
                )
                self.send_message(conf_wgt_msg)

            self.running.emit()

            for idx, job in enumerate(jobs):
                if self.should_exit():
                    self.handle_exit(pool)
                    return

                results.append(job.get())

                self.report_progress(
                    SIGNAL_OWNER.GENERATE_MRI_DATASET_WORKER,
                    JOB_TYPE.GENERATE_MRI_DATASET,
                    idx,
                    len(jobs),
                )

        for r in results:
            if r is not None:
                df = df.append(r, ignore_index=True)

        logger.info('Generating MRI dataset finished.')

        logger.info(f'Saving metadata to file: {str(metadata_csv_file)}.')
        df.to_csv(metadata_csv_file)

        if dfdc_fract < 1.0:
            df, _ = train_test_split(df, train_size=dfdc_fract)

        # convert ['real_image', 'fake_image', 'mri_image'] to
        # ['face_image', 'mri_image']
        # where if image is real => use blank image as mri
        #       else use the mri(real_image, fake_image)

        dff = pd.DataFrame(columns=['face_image', 'mri_image', 'class'])
        dff['face_image'] = df['fake_image'].unique()
        dff['mri_image'] = df['mri_image']
        dff_len = len(dff)
        dff['class'][0:dff_len] = 'fake'
        dff = dff.set_index('face_image')

        dfr_base = pd.DataFrame(columns=['face_image', 'mri_image', 'class'])
        dfr_base['face_image'] = df['real_image'].unique()
        dfr_base['mri_image'] = self._get_blank_image_path().absolute()
        dfr_base_len = len(dfr_base)
        dfr_base['class'][0:dfr_base_len] = 'real'

        dfr = dfr_base
        dfr_len = len(dfr)
        dfr = dfr.set_index('face_image')

        real_train, real_test = train_test_split(dfr, test_size=test_size)
        mri_train_real_csv_path = self._get_mri_train_real_dataset_csv_path()
        logger.info(
            f'Saving mri train real dataset csv to ' +
            f'file: {str(mri_train_real_csv_path)}.'
        )
        real_train.to_csv(mri_train_real_csv_path)
        mri_test_real_csv_path = self._get_mri_test_real_dataset_csv_path()
        logger.info(
            f'Saving mri test real dataset csv to ' +
            f'file: {str(mri_test_real_csv_path)}.'
        )
        real_test.to_csv(mri_test_real_csv_path)

        fake_train, fake_test = train_test_split(dff, test_size=test_size)
        mri_train_fake_csv_path = self._get_mri_train_fake_dataset_csv_path()
        logger.info(
            f'Saving mri train fake dataset csv to ' +
            f'file: {str(mri_train_fake_csv_path)}.'
        )
        fake_train.to_csv(mri_train_fake_csv_path)
        mri_test_fake_csv_path = self._get_mri_test_fake_dataset_csv_path()
        logger.info(
            f'Saving mri test fake dataset csv to ' +
            f'file: {str(mri_test_fake_csv_path)}.'
        )
        fake_test.to_csv(mri_test_fake_csv_path)

        total_samples = dfr_len + dff_len

        logger.info(f'Total fake samples {dff_len}')
        logger.info(f'Total real samples {dfr_len}')
        logger.info(f'Fake train samples {len(fake_train)}')
        logger.info(f'Real train samples {len(real_train)}')
        logger.info(f'Fake test samples {len(fake_test)}')
        logger.info(f'Real test samples {len(real_test)}')
        logger.info(
            f'Total samples {total_samples}, ' +
            f'real={round((dfr_len / total_samples) * 100, 2)}% ' +
            f'fake={round((dff_len / total_samples) * 100, 2)}%'
        )
