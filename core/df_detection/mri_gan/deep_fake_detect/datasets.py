import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from configs.mri_gan_config import MRIGANConfig

from core.df_detection.mri_gan.utils import ConfigParser
from enums import MODE, MRI_GAN_DATASET


class DFDCDatasetSimple(Dataset):

    def __init__(
        self,
        mode: MODE,
        transform: transforms.Compose,
        data_size: int,
        dataset: MRI_GAN_DATASET,
        label_smoothing: float = 0.,
    ):
        super().__init__()
        self.mode = mode
        # use only in training, so update to param passed in train mode
        self.label_smoothing = 0
        if mode == MODE.TRAIN:
            if dataset == MRI_GAN_DATASET.PLAIN:
                self.labels_csv = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_dfdc_train_frame_label_csv_path()
                )
                self.crops_dir = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_dfdc_crops_train_path()
                )
            elif dataset == MRI_GAN_DATASET.MRI:
                self.labels_csv = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_train_mriframe_label_csv_path()
                )
                self.crops_dir = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_train_mrip2p_png_data_path()
                )
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')

            self.label_smoothing = label_smoothing
        elif mode == MODE.VALID:
            if dataset == MRI_GAN_DATASET.PLAIN:
                self.labels_csv = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_dfdc_valid_frame_label_csv_path()
                )
                self.crops_dir = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_dfdc_crops_valid_path()
                )
            elif dataset == MRI_GAN_DATASET.MRI:
                self.labels_csv = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_valid_mriframe_label_csv_path()
                )
                self.crops_dir = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_valid_mrip2p_png_data_path()
                )
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')

        elif mode == MODE.TEST:
            if dataset == MRI_GAN_DATASET.PLAIN:
                self.labels_csv = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_dfdc_test_frame_label_csv_path()
                )
                self.crops_dir = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_dfdc_crops_test_path()
                )
            elif dataset == MRI_GAN_DATASET.MRI:
                self.labels_csv = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_test_mriframe_label_csv_path()
                )
                self.crops_dir = Path(
                    MRIGANConfig
                    .get_instance()
                    .get_test_mrip2p_png_data_path()
                )
            else:
                raise Exception('Bad dataset in DFDCDatasetSimple')
        else:
            raise Exception('Bad mode in DFDCDatasetSimple')
        self.data_df = pd.read_csv(self.labels_csv)
        if data_size < 1:
            total_data_len = int(len(self.data_df) * data_size)
            self.data_df = self.data_df.iloc[0:total_data_len]
        self.data_dict = self.data_df.to_dict(orient='records')
        self.data_len = len(self.data_df)
        self.transform = transform

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        while True:
            try:
                item = self.data_dict[index].copy()
                part, video_id, frame_name, label = item.values()
                frame_path = self.crops_dir / part / video_id / frame_name
                frame = Image.open(frame_path)
                if self.transform is not None:
                    frame = self.transform(frame)
                item['frame_tensor'] = frame
                if self.label_smoothing != 0:
                    label = np.clip(
                        label,
                        self.label_smoothing,
                        1 - self.label_smoothing,
                    )
                item['label'] = torch.tensor(label)
                return item
            except Exception as e:
                index = random.randint(0, self.data_len)
