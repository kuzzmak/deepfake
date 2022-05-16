import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import PyQt6.QtCore as qtc

from core.df_detection.mri_gan.data_utils.utils import (
    filter_dfdc_dirs,
    get_metadata_file_paths,
)
from core.worker import MRIGANWorker, WorkerWithPool
from enums import (
    DATA_TYPE,
    JOB_NAME,
    JOB_TYPE,
    MRI_GAN_DATASET,
    SIGNAL_OWNER,
    WIDGET,
)
from message.message import Messages

logger = logging.getLogger(__name__)


class GenerateFrameLabelsCSVWorker(MRIGANWorker, WorkerWithPool):
    """Worker for generating frame labels CSV files.

    Parameters
    ----------
    data_type : DATA_TYPE
        for which data type CSV files are being generated
    mri_gan_dataset : MRI_GAN_DATASET
        for which MRI GAN dataset are files being generated
    num_instances : int, optional
        how many of the same worker processes will be launched, by default 2
    message_worker_sig : Optional[qtc.pyqtSignal], optional
        signal to the message worker, by default None
    """

    def __init__(
        self,
        data_type: DATA_TYPE,
        mri_gan_dataset: MRI_GAN_DATASET,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        MRIGANWorker.__init__(self, data_type)
        WorkerWithPool.__init__(self, num_instances, message_worker_sig)

        self._mri_gan_dataset = mri_gan_dataset

    def _get_training_reals_and_fakes(self) -> Tuple[List[str], List[str]]:
        """Constructs lists of fake and original filenames based on the
        metadata.json files that come with the datasets for the training
        dataset.

        Returns
        -------
        Tuple[List[str], List[str]]
            list of original filenames, list of fake filenames
        """
        root_dir = self._get_dfdc_data_path()
        metadata_paths = get_metadata_file_paths(root_dir)
        originals = []
        fakes = []
        for json_path in metadata_paths:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            for k, v in metadata.items():
                if v['label'] == 'FAKE':
                    fakes.append(k)
                else:
                    originals.append(k)
        return originals, fakes

    def _get_valid_or_test_reals_and_fakes(
        self
    ) -> Tuple[List[str], List[str]]:
        """Constructs lists of fake and original filenames based on the
        labels.csv files that come with the datastes for valid and test
        datasets.

        Returns
        -------
        Tuple[List[str], List[str]]
            list of original filenames, list of fake filenames
        """
        labels_csv = self._get_dfdf_data_label_csv_path()
        df = pd.read_csv(labels_csv, index_col=0)
        originals = list(df[df['label'] == 0].index.values)
        fakes = list(df[df['label'] == 1].index.values)
        return originals, fakes

    @staticmethod
    def _get_video_frame_labels_mapping(c_id_path: Path, originals, fakes):
        c_id = c_id_path.stem
        if c_id in originals:
            crop_label = 0
        elif c_id in fakes:
            crop_label = 1
        else:
            raise Exception('Unknown label')

        df = pd.DataFrame(columns=['part', 'video_id', 'frame', 'label'])
        for crp_itm in list(c_id_path.glob('*.*')):
            new_row = {
                'part': c_id_path.parent.name,
                'video_id': c_id,
                'frame': crp_itm.name,
                'label': crop_label,
            }
            df = df.append(new_row, ignore_index=True)
        return df

    def _get_data_reals_and_fakes(self):
        """Constructs lists of fake and original filenames based on the
        files that com with the datastes.

        Returns
        -------
        Tuple[List[str], List[str]]
            list of original filenames, list of fake filenames
        """
        if self._data_type == DATA_TYPE.TRAIN:
            originals, fakes = self._get_training_reals_and_fakes()
        else:
            originals, fakes = self._get_valid_or_test_reals_and_fakes()
        return originals, fakes

    def run_job(self) -> None:
        logger.info(
            'Generation of frame labels CSV for ' +
            f'{self._data_type.value} dataset started.'
        )
        # read which videos are reald and which ones are fake
        # training part of the dataset reads this from metadata file but valid
        # and test dataset read this from labels.csv file
        originals_, fakes_ = self._get_data_reals_and_fakes()
        if self._mri_gan_dataset == MRI_GAN_DATASET.PLAIN:
            csv_file = self._get_dfdc_frame_label_csv_path()
            crop_path = self._get_dfdc_crops_data_path()
        elif self._mri_gan_dataset == MRI_GAN_DATASET.MRI:
            csv_file = self._get_mriframe_label_csv_path()
            crop_path = self._get_mrip2p_png_data_path()
        else:
            raise Exception('Bad dataset')

        # get only video names with no extension
        originals = set(
            [
                os.path.splitext(video_filename)[0]
                for video_filename in originals_
            ]
        )
        fakes = set(
            [
                os.path.splitext(video_filename)[0]
                for video_filename in fakes_
            ]
        )

        dfdc_dirs = filter_dfdc_dirs(os.listdir(crop_path))
        crop_id_paths = []
        [
            crop_id_paths.extend(
                [crop_path / d / id for id in os.listdir(crop_path / d)]
            )
            for d in dfdc_dirs
        ]

        logger.info(f'Found {len(crop_id_paths)} video directories.')

        with multiprocessing.Pool(self._num_instances) as pool:
            jobs = []
            for c_id_path in crop_id_paths:
                jobs.append(
                    pool.apply_async(
                        GenerateFrameLabelsCSVWorker.
                        _get_video_frame_labels_mapping,
                        (c_id_path, originals, fakes,),
                    )
                )
                break

            conf_wgt_msg = Messages.CONFIGURE_WIDGET(
                SIGNAL_OWNER.GENERATE_FRAME_LABELS_CSV_WORKER,
                WIDGET.JOB_PROGRESS,
                'setMaximum',
                [len(jobs)],
                JOB_NAME.GENERATE_FRAME_LABELS_CSV,
            )
            self.send_message(conf_wgt_msg)

            self.running.emit()

            results = []
            for idx, job in enumerate(jobs):
                if self.should_exit():
                    self.handle_exit(pool)
                    break

                results.append(job.get())

                self.report_progress(
                    SIGNAL_OWNER.GENERATE_FRAME_LABELS_CSV_WORKER,
                    JOB_TYPE.GENERATE_FRAME_LABELS_CSV,
                    idx,
                    len(jobs),
                )

        df = pd.DataFrame(columns=['part', 'video_id', 'frame', 'label'])
        for r in results:
            df = df.append(r, ignore_index=True)
        df.set_index(['part', 'video_id'], inplace=True)
        logger.debug(
            f'Saving frame labels for {self._data_type.value} ' +
            f'dataset to {str(csv_file)}.'
        )
        df.to_csv(csv_file)
        logger.debug('Frame label file saved.')

        logger.info(
            'Generation of frame labels CSV for ' +
            f'{self._data_type.value} dataset finished.'
        )
