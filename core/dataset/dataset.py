import logging
from typing import Callable, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
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
        output_shape: int,
        transformations: Optional[nn.Module] = None,
        image_augmentations: List[Callable] = [],
        device: DEVICE = DEVICE.CPU,
    ):
        """Constructor.

        Parameters
        ----------
        metadata_path_A : str
            path of the `Faces` metadata of person A
        metadata_path_B : str
            path of the `Faces` metadata of person B
        input_shape : int
            size of the square to which input face will be resized for the
                model input
        output_shape : int
            size of the square to which output will be resized that represents
                models target
        transformations : Optional[torch.Module]
            transformations for the output of the dataset like converting
                numpy arrays to torch tensors
        image_augmentations : List[Callable]
            list of functions for doing augmentations on image
        device : DEVICE, optional
            where to send loaded faces and masks, by default DEVICE.CPU
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
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

    def _resize(
        self,
        warped_A: np.ndarray,
        mask_A: np.ndarray,
        target_A: np.ndarray,
        warped_B: np.ndarray,
        mask_B: np.ndarray,
        target_B: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Resizes input arrays so they can be used for model input and error
        calculation when model makes forward pass. `warped_A` and `warped_B`
        are resized to the model input and other ones to the model output size.

        Args:
            warped_A (np.ndarray): warped image input for person A
            mask_A (np.ndarray): mask for person A
            target_A (np.ndarray): target image for person A
            warped_B (np.ndarray): warped image input for person B
            mask_B (np.ndarray): mask for person B
            target_B (np.ndarray): target image for person B

        Returns:
            Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, ]: resized input arrays
        """
        input_shape = (self.input_shape, self.input_shape)
        output_shape = (self.output_shape, self.output_shape)
        return (
            cv.resize(warped_A, input_shape),
            cv.resize(mask_A, output_shape),
            cv.resize(target_A, output_shape),
            cv.resize(warped_B, input_shape),
            cv.resize(mask_B, output_shape),
            cv.resize(target_B, output_shape),
        )

    def _transform(
        self,
        warped_A: np.ndarray,
        mask_A: np.ndarray,
        target_A: np.ndarray,
        warped_B: np.ndarray,
        mask_B: np.ndarray,
        target_B: np.ndarray,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Makes transformations on the input arrays. Transformation to torch
        tensor and similar.

        Args:
            warped_A (np.ndarray): warped image input for person A
            mask_A (np.ndarray): mask for person A
            target_A (np.ndarray): target image for person A
            warped_B (np.ndarray): warped image input for person B
            mask_B (np.ndarray): mask for person B
            target_B (np.ndarray): target image for person B

        Returns:
            Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, ]: transformed input arrays
        """
        return (
            self.transformations(warped_A),
            self.transformations(mask_A),
            self.transformations(target_A),
            self.transformations(warped_B),
            self.transformations(mask_B),
            self.transformations(target_B)
        )

    def __getitem__(self, index: int) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        face_A = self.A_faces[index]
        face_B = self.B_faces[self._similarity_indices[index]]
        target_A = face_A.aligned_image
        target_B = face_B.aligned_image
        warped_A, warped_B = ImageAugmentation.warp_faces(
            cv.INTER_LINEAR,
            face_A,
            face_B,
        )
        return self._transform(
            *self._resize(
                warped_A,
                face_A.aligned_mask,
                target_A,
                warped_B,
                face_B.aligned_mask,
                target_B,
            )
        )
