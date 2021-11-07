import logging
from typing import Callable, List, Optional, Tuple

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
        metadata_path_A: str,
        metadata_path_B: str,
        input_shape: int,
        image_augmentations: List[Callable] = [],
        load_into_memory: bool = False,
        device: DEVICE = DEVICE.CPU,
        transforms: Optional[transforms.Compose] = None,
    ):
        """Constructor.

        Parameters
        ----------
        metadata_path_A : str
            path of the `Faces` metadata of person A
        metadata_path_B : str
            path of the `Faces` metadata of person B
        input_shape : int
            size of the square to which face and mask are resized
        image_augmentations : List[Callable]
            list of functions for doing augmentations on image
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
        self.image_augmentations = image_augmentations
        self.transforms = transforms if transforms is not None \
            else default_transforms
        self.metadata_paths_A = get_file_paths_from_dir(metadata_path_A, ['p'])
        self.metadata_paths_B = get_file_paths_from_dir(metadata_path_B, ['p'])
        if self.load_into_memory:
            self._load()

    def _load(self):
        """Loads dataset into ram or gpu.
        """
        self.faces_A = []
        self.masks_A = []
        self.faces_B = []
        self.masks_B = []
        logger.info(
            'Loading dataset into memory ' +
            f"({'GPU' if self.device == DEVICE.CUDA else 'RAM'})."
        )
        for path in self.metadata_paths_A:
            face_A, mask_A = self._load_from_path(path)
            self.faces_A.append(face_A)
            self.masks_A.append(mask_A)
        for path in self.metadata_paths_B:
            face_B, mask_B = self._load_from_path(path)
            self.faces_B.append(face_B)
            self.masks_B.append(mask_B)
        logger.info(
            f'Loaded {len(self.faces_A)} face_A metadata ' +
            f'and {len(self.faces_B)} face_B metadata into memory.'
        )

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

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Augments every image in the batch based on the image augmentation
        function and parameters user chose.

        Args:
            image (np.ndarray): image to be augmented

        Returns:
            np.ndarray: augmented image if augmentations were chosen, normal
                image otherwise
        """
        # for aug in self.image_augmentations:
        #     image = aug(image)
        return image

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
        return min(len(self.metadata_paths_A), len(self.metadata_paths_B))

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.load_into_memory:
            # load from memory
            return (
                self._augment(self.faces_A[index]),
                self.faces_A[index],
                self.masks_A[index],

                self._augment(self.faces_B[index]),
                self.faces_B[index],
                self.masks_B[index],
            )
        # load from disk
        path_A = self.metadata_paths_A[index]
        path_B = self.metadata_paths_B[index]
        face_A, mask_A = self._load_from_path(path_A)
        face_B, mask_B = self._load_from_path(path_B)
        return (
            self._augment(face_A),
            face_A,
            mask_A,
            self._augment(face_B),
            face_B,
            mask_B,
        )
