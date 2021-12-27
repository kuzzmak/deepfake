import logging
from typing import Callable, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from core.face_alignment.face_aligner import FaceAligner
from core.image.augmentation import ImageAugmentation
from enums import DEVICE
from serializer.face_serializer import FaceSerializer
from utils import get_file_paths_from_dir

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
        device: DEVICE = DEVICE.CPU,
        transformations: Optional[transforms.Compose] = None,
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
        device : DEVICE, optional
            where to send loaded faces and masks, by default DEVICE.CPU
        transformations : Optional[transforms.Compose], optional
            transformations for the dataset, by default None
        """
        self.input_shape = input_shape
        self.device = device
        self.image_augmentations = image_augmentations
        self.transformations = transformations if transformations is not None \
            else transforms.Compose([transforms.ToTensor()])
        self.metadata_paths_A = get_file_paths_from_dir(metadata_path_A, ['p'])
        self.metadata_paths_B = get_file_paths_from_dir(metadata_path_B, ['p'])
        self._similarity_indices = dict()
        self._load()
        self._align()
        self._find_nearest_faces()

    def _load(self):
        """Loads dataset into memory.
        """
        logger.info('Loading faces A, please wait...')
        self.A_faces = [FaceSerializer.load(path)
                        for path in self.metadata_paths_A]
        logger.info(f'Loaded {len(self.A_faces)} faces A.')
        logger.info('Loading faces B, please wait...')
        self.B_faces = [FaceSerializer.load(path)
                        for path in self.metadata_paths_B]
        logger.info(f'Loaded {len(self.B_faces)} faces B.')

    def _align(self) -> None:
        """Aligns face to the mean face i.e. aligned face, landmarks and
        mask are resized to the `input_shape`.
        """
        logger.info('Aligning faces A, please wait...')
        [FaceAligner.align_face(face, self.input_shape)
         for face in self.A_faces]
        logger.info('Aligned faces A.')
        logger.info('Aligning faces B, please wait...')
        [FaceAligner.align_face(face, self.input_shape)
         for face in self.B_faces]
        logger.info('Aligned faces B.')

    def _find_nearest_faces(self) -> None:
        """Function which constructs a dictionary where each key is the ordinal
        number of the face A in the list of A faces and the value is the
        ordinal number of the face in list of B faces that is most similar to
        the face A.
        """
        logger.info('Finding nearest faces, please wait...')
        b_landmarks = [f.aligned_landmarks for f in self.B_faces]
        for i, f in enumerate(self.A_faces):
            diff = np.average(
                np.square(f.aligned_landmarks - b_landmarks),
                axis=(1, 2),
            )
            best_indices = diff.argsort()
            self._similarity_indices[i] = best_indices[0]
        logger.info('Finding nearest faces done.')

    def __len__(self):
        return len(self.A_faces)

    def __getitem__(self, index: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        face_A = self.A_faces[index]
        face_B = self.B_faces[self._similarity_indices[index]]
        target_A = face_A.aligned_image
        target_B = face_B.aligned_image
        warped_A, warped_mask_A, warped_B, warped_mask_B = \
            ImageAugmentation.warp_faces_and_masks(
                cv.INTER_LINEAR,
                face_A,
                face_B,
            )
        return (
            warped_A,
            warped_mask_A,
            target_A,
            warped_B,
            warped_mask_B,
            target_B,
        )
