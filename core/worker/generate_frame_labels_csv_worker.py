import json
import multiprocessing
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import PyQt6.QtCore as qtc

from core.df_detection.mri_gan.data_utils.utils import (
    filter_dfdc_dirs, 
    get_metadata_file_paths,
)
from core.worker import MRIGANWorker, WorkerWithPool
from enums import DATA_TYPE


class GenerateFrameLabelsCSVWorker(MRIGANWorker, WorkerWithPool):

    def __init__(
        self,
        data_type: DATA_TYPE,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        MRIGANWorker.__init__(self, data_type)
        WorkerWithPool.__init__(self, num_instances, message_worker_sig)

    def _get_training_reals_and_fakes(self):
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

    @staticmethod
    def _get_video_frame_labels_mapping(c_id_path: Path, originals, fakes):
        c_id = c_id_path.stem
        if c_id in originals:
            crop_label = 0
        elif c_id in fakes:
            crop_label = 1
        else:
            raise Exception('Unknown label')

        df = pd.DataFrame(columns=['video_id', 'frame', 'label'])
        for crp_itm in list(c_id_path.glob('*.*')):
            new_row = {
                'video_id': c_id,
                'frame': crp_itm.stem,
                'label': crop_label,
            }
            df = df.append(new_row, ignore_index=True)
        return df

    def run_job(self) -> None:
        dataset = 'plain'
        originals_, fakes_ = self._get_training_reals_and_fakes()
        if dataset == 'plain':
            csv_file = self._get_dfdc_frame_label_csv_path()
            crop_path = self._get_dfdc_crops_data_path()
        elif dataset == 'mri':
            csv_file = self._get_mriframe_label_csv_path()
            crop_path = self._get_mrip2p_png_data_path()
        else:
            raise Exception('Bad dataset')

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

            results = []
            for job in jobs:
                results.append(job.get())

        df = pd.DataFrame(columns=['video_id', 'frame', 'label'])
        for r in results:
            df = df.append(r, ignore_index=True)
        df.set_index('video_id', inplace=True)
        df.to_csv(csv_file)
