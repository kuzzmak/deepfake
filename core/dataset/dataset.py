import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from enums import DEVICE
from serializer.face_serializer import FaceSerializer
from utils import get_file_paths_from_dir

# if no transformation are passed on class initialization, this default
# transformation is used
default_transforms = transforms.Compose([transforms.ToTensor()])

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """Deepfake dataset class containing detected faces and masks.
    """

    def __init__(
        self,
        metadata_path: str,
        input_shape: int,
        load_into_memory: bool = False,
        device: DEVICE = DEVICE.CPU,
        transforms: Optional[transforms.Compose] = None,
    ):
        """Constructor.

        Parameters
        ----------
        metadata_path : str
            path of the `Faces` metadata
        input_shape : int
            size of the square to which face and mask are resized
        load_into_memory : bool, optional
            should dataset be loaded into memory, by default False
        device : DEVICE, optional
            where to send loaded faces and masks, by default DEVICE.CPU
        transforms : Optional[transforms.Compose], optional
            transformations for the dataset, by default None
        """
        self.input_shape = input_shape
        self.load_into_memory = load_into_memory
        self.device = device
        self.transforms = transforms if transforms is not None \
            else default_transforms
        self.metadata_paths = get_file_paths_from_dir(metadata_path, ['p'])
        if self.load_into_memory:
            logger.info(
                f"Loading dataset into memory ({'GPU' if device == DEVICE.CUDA else 'RAM'}).")
            self._load()
            logger.info(
                f'Loaded: {len(self.faces)} face metadata into memory.'
            )

    def _load(self):
        """Loads dataset into ram or gpu.
        """
        self.faces = []
        self.masks = []
        for path in self.metadata_paths:
            face, mask = self._load_from_path(path)
            self.faces.append(face.to(self.device.value))
            self.masks.append(mask.to(self.device.value))

    def _load_from_path(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads face and mask from `Face` metadata. They are then aligned and
        transformed if some kind of transformations were passed as an argument,
        else default transformation (to vector) is done.

        Parameters
        ----------
        path : str
            path of the `Face` metadata

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            face and mask tensors
        """
        face = FaceSerializer.load(path)
        aligned_face, aligned_mask = self._align(face)
        aligned_face, aligned_mask = self._transform(
            aligned_face,
            aligned_mask,
        )
        aligned_face = aligned_face.to(self.device.value)
        aligned_mask = aligned_mask.to(self.device.value)
        return aligned_face, aligned_mask

    def _align(self, face: Face) -> Tuple[np.ndarray, np.ndarray]:
        """Aligns face and mask with the help of the `FaceAligner` class.

        Parameters
        ----------
        face : Face
            `Face` object containing face and mask

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            aligned face and mask
        """
        aligned_face = FaceAligner.get_aligned_face(face, self.input_shape)
        aligned_mask = FaceAligner.get_aligned_mask(face, self.input_shape)
        return aligned_face, aligned_mask

    def _transform(
        self,
        face: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms face and mask which are passed as an argument based on
        the transformations of this class.

        Parameters
        ----------
        face : np.ndarray
            face array
        mask : np.ndarray
            mask array

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            transformed face and mask tensor
        """
        face = self.transforms(face)
        mask = self.transforms(mask)
        return face, mask

    def __len__(self):
        return len(self.metadata_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.load_into_memory:
            # laod from memory
            return self.faces[index], self.faces[index], self.masks[index]
        # laod from disk
        path = self.metadata_paths[index]
        face, mask = self._load_from_path(path)
        return face, face, mask
