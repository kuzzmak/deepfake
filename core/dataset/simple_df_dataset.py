from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from core.aligner import Aligner
from core.dictionary import Dictionary
from serializer.face_serializer import FaceSerializer
from utils import (
    get_aligned_landmarks_filename,
    get_file_paths_from_dir, prepare_path,
)


class SimpleDFDataset(Dataset):

    def __init__(self, data_path: Union[str, Path], input_size: int) -> None:
        super().__init__()

        self._input_size = input_size
        self._transformations = transforms.Compose([transforms.ToTensor()])

        data_path = prepare_path(data_path)
        if data_path is None:
            raise Exception('Provided path for data doesn\'t exist.')

        self._data_paths = get_file_paths_from_dir(data_path, ['p'])

        landmarks_file = get_aligned_landmarks_filename(input_size)
        landmarks_path = data_path / landmarks_file
        self._landmarks = Dictionary.load(landmarks_path)

        alignments_file = f'alignments.json'
        alignments_path = data_path / alignments_file
        self._alignments = Dictionary.load(alignments_path)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self._data_paths[index]
        face = FaceSerializer.load(path)
        img = face.raw_image.data
        aligned = Aligner.align_image(
            img,
            self._alignments[path.name],
            self._input_size,
        )
        return self._transformations(aligned)

    def __len__(self) -> int:
        return len(self._data_paths)
