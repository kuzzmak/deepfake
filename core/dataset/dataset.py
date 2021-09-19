from core.face_alignment.face_aligner import FaceAligner
from core.face import Face
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from serializer.face_serializer import FaceSerializer
from utils import get_file_paths_from_dir


class DeepfakeDataset(Dataset):

    def __init__(
        self,
        faces_path: str,
        input_shape: int,
        transforms: Optional[transforms.Compose] = None,
    ):
        paths = get_file_paths_from_dir(faces_path, ['p'])
        self.faces_path = paths
        self.input_shape = input_shape
        self.transforms = transforms

    def __len__(self):
        return len(self.faces_path)

    def __getitem__(self, index: int) -> Face:
        face_path = self.faces_path[index]
        face = FaceSerializer.load(face_path)
        aligned = FaceAligner.get_aligned_face(face, self.input_shape)
        if self.transforms:
            aligned = self.transforms(aligned)
        # mask = face.mask
        # if self.transforms:
        #     print('in transform')
        #     print(detected_face.shape)
        #     print(mask.shape)
        #     detected_face = self.transforms(detected_face)
        #     mask = self.transforms(mask)
        # return detected_face, mask
        return aligned
