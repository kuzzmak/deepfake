import logging
import random
from operator import itemgetter
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from core.aligner import Aligner
from core.dictionary import Dictionary
from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from core.face_alignment.utils import get_face_mask
from core.image.augmentation import ImageAugmentation
from serializer.face_serializer import FaceSerializer
from utils import (get_aligned_landmarks_filename, get_file_paths_from_dir,
                   get_image_paths_from_dir)

logger = logging.getLogger(__name__)

"""Constructor.

Parameters
----------
path_A : Union[str, Path]
    path of the aligned images for person A
path_B : Union[str, Path]
    path of the aligned images for person B
nearest_n : int
    how many nearest images to consider when selecting target face,
        difference is measured as a MSE between face landmarks,
        by default 10
transformations : Optional[torch.Module]
    transformations for the output of the dataset like converting
        numpy arrays to torch tensors
image_augmentations : List[Callable]
    list of functions for doing augmentations on image
"""


class DeepfakeDataset(Dataset):
    """Deepfake dataset class containing detected faces and masks.

    Args:
        path_A (Union[str, Path]): path to the directory with metadata files
            for person A
        path_B (Union[str, Path]): path to the directory with metadata files
            for person B
        input_size (int): size of the square which model receives on input
        output_size (int): size of the square which model receives on output
        nearest_n (int, optional): how many nearest faces in B should be found
            for every face in A. Defaults to 10.
        transformations (Optional[nn.Module], optional): transformations which
            will be applied to every image model receives on input or output.
            Defaults to None.
        image_augmentations (List[Callable], optional): augmentation functions
            which will be applied on every image model receives on input.
            Defaults to [].
    """

    def __init__(
        self,
        path_A: Union[str, Path],
        path_B: Union[str, Path],
        input_size: int,
        output_size: int,
        nearest_n: int = 10,
        transformations: Optional[nn.Module] = None,
        image_augmentations: List[Callable] = [],
    ):
        if isinstance(path_A, str):
            path_A = Path(path_A)
        if isinstance(path_B, str):
            path_B = Path(path_B)
        self._path_A = path_A
        self._path_B = path_B
        self._input_size = input_size
        self._output_size = output_size
        self._nearest_n = nearest_n
        self._image_augmentations = image_augmentations
        self._transformations = transformations \
            if transformations is not None \
            else transforms.Compose([transforms.ToTensor()])

        self._paths_A = []
        self._paths_B = []
        self._landmarks_A = None
        self._landmarks_B = None
        self._alignments_A = None
        self._alignments_B = None
        self._nearest_n_dict = dict()

        self._load_paths()
        self._load_landmarks()
        self._load_alignments()
        self._find_nearest_n()

    def _load_paths(self) -> None:
        """Generates file paths for both A and B metadata files.
        """
        self._paths_A = get_file_paths_from_dir(self._path_A, ['p'])
        logger.info(
            f'Found {len(self._paths_A)} images for person A in ' +
            f'directory {str(self._path_A)}.'
        )
        self._paths_B = get_file_paths_from_dir(self._path_B, ['p'])
        logger.info(
            f'Found {len(self._paths_B)} images for person B in ' +
            f'directory {str(self._path_B)}.'
        )

    def _load_landmarks(self) -> None:
        """Loads aligned_landmarks.json file for both A and B persons.
        """
        landmarks_file = get_aligned_landmarks_filename(self._input_size)
        logger.debug(f'Loading {landmarks_file} file for person A.')
        landmarks_path_A = self._path_A / landmarks_file
        if not landmarks_path_A.exists():
            logger.error(
                f'{landmarks_file} file for person A ' +
                f'does not exist on path {str(landmarks_path_A)}'
            )
        else:
            self._landmarks_A = Dictionary.load(landmarks_path_A)
            self._size = len(self._landmarks_A)
            logger.debug(
                f'Successfully loaded {landmarks_file} file for person A.'
            )

        logger.debug(f'Loading {landmarks_file} file for person B.')
        landmarks_path_B = self._path_B / landmarks_file
        if not landmarks_path_B.exists():
            logger.error(
                f'{landmarks_file} file for person A ' +
                f'does not exist on path {str(landmarks_path_B)}'
            )
        else:
            self._landmarks_B = Dictionary.load(landmarks_path_B)
            logger.debug(
                f'Successfully loaded {landmarks_file} file for person B.'
            )

    def _load_alignments(self) -> None:
        """Loads alignment.json files for both A and B persons.
        """
        alignments_file = f'alignments.json'
        logger.debug(f'Loading {alignments_file} file for person A.')
        alignments_path_A = self._path_A / alignments_file
        if not alignments_path_A.exists():
            logger.error(
                f'{alignments_file} file for person A ' +
                f'does not exist on path {str(alignments_path_A)}'
            )
        else:
            self._alignments_A = Dictionary.load(alignments_path_A)
            logger.debug(
                f'Successfully loaded {alignments_file} file for person A.'
            )

        logger.debug(f'Loading {alignments_file} file for person B.')
        alignments_path_B = self._path_B / alignments_file
        if not alignments_path_B.exists():
            logger.error(
                f'{alignments_file} file for person B ' +
                f'does not exist on path {str(alignments_path_B)}'
            )
        else:
            self._alignments_B = Dictionary.load(alignments_path_B)
            logger.debug(
                f'Successfully loaded {alignments_file} file for person B.'
            )

    def _find_nearest_n(self) -> None:
        """Finds nearest B faces for every A face based on face landmarks.
        Number of faces depends on the `nearest_n` argument.
        """
        logger.debug('Finding nearest faces.')
        # keys are metadata file names
        keys_A = self._landmarks_A.keys()
        keys_B = list(self._landmarks_B.keys())
        # values are aligned landmarks
        values_A = list(self._landmarks_A.values())
        values_B = np.array(list(self._landmarks_B.values()))
        # dictionary where key is the file name and value is a list of
        # nearest_n nearest aligned faces from face B
        nearest_n_A = dict()
        for i, im_path in enumerate(keys_A):
            curr = values_A[i]
            # MSE of the current landmarks and all other from face B
            diff = np.average(
                np.square(curr - values_B),
                axis=(1, 2),
            )
            # get nearest_n closest face landmarks
            best_n = diff.argsort()[:self._nearest_n]
            best_n_paths = itemgetter(*best_n)(keys_B)
            if not isinstance(best_n_paths, tuple):
                best_n_paths = (best_n_paths,)
            nearest_n_A[im_path] = best_n_paths
        self._nearest_n_dict = nearest_n_A
        logger.debug('Finding nearest faces finished.')

    def __len__(self):
        return len(self._paths_A)

    def _prepare_image(
        self,
        path: Path,
        alignment: np.ndarray,
        landmarks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads `Face` metadata object from `path` and transformes detected
        face based on the `alignment` matrix to model input image, target image
        and mask for calculating error.

        Args:
            path (Path): path to `Face` metadata object
            alignment (np.ndarray): alignment matrix for this face
            landmarks (np.ndarray): aligned landmarks for mask generation

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: warped face image,
                target face image, input face mask
        """
        face = FaceSerializer.load(path)
        img = face.raw_image.data
        aligned = Aligner.align_image(img, alignment, self._input_size)
        warped = ImageAugmentation.warp_image(cv.INTER_CUBIC, aligned)
        target = Aligner.align_image(img, alignment, self._output_size)
        mask = get_face_mask(aligned, landmarks)
        mask = cv.resize(mask, (self._output_size, self._output_size))
        return warped, mask, target,

    def _transform(
        self,
        images: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """Transforms every image in input list based on the transformations
        passed on `DeepfakeDataset` init. Makes input image for model, output
        image for model and mask for the output image.

        Args:
            images (List[np.ndarray]): images to transform

        Returns:
            List[torch.Tensor]: tensor list of transformed images
        """
        return list(map(lambda im: self._transformations(im), images))

    def __getitem__(self, index) -> List[np.ndarray]:
        path_A = self._paths_A[index]
        nearest = self._nearest_n_dict[path_A.name]
        random_nearest_B: str = random.choice(nearest)
        path_B = self._path_B / random_nearest_B
        return self._transform([
            *self._prepare_image(
                path_A,
                self._alignments_A[path_A.name],
                self._landmarks_A[path_A.name],
            ),
            *self._prepare_image(
                path_B,
                self._alignments_B[path_B.name],
                self._landmarks_B[path_B.name],
            )
        ])
