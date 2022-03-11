from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from torchvision import transforms


@dataclass
class DatasetConfiguration:
    """Configuration class for the `Deepfake` dataset.

    Parameters
    ----------
    path_A : Path
        path of the directory containing `Face` metadata of person A
    path_B : Path
        path of the directory containing `Face` metadata of person B
    batch_size : int
        batch size
    input_size : int
        size of the square image on the model input
    output_size : int
        size of the square image on the model output
    image_augmentaions : List[Callable]
        list of functions for image augmentation
    data_transforms : Optional[transforms.Compose], optional
        transformations for the dataset, by default None
    shuffle : bool
        should the dataset be shuffled, by default True
    num_workers : int
        number of threads used for dataset loading, by default 2
    """

    def __init__(
        self,
        path_A: Union[str, Path],
        path_B: Union[str, Path],
        input_size: int,
        output_size: int,
        batch_size: int,
        image_augmentations: List[Callable] = field(),
        data_transforms: Optional[transforms.Compose] = None,
        shuffle: bool = True,
        num_workers: int = 2,
    ) -> None:
        if isinstance(path_A, str):
            path_A = Path(path_A)
        self._path_A = path_A
        if isinstance(path_B, str):
            path_B = Path(path_B)
        self._path_B = path_B
        self._input_size = input_size
        self._output_size = output_size
        self._batch_size = batch_size
        self._image_augmentations = image_augmentations
        self._data_transforms = data_transforms
        self._shuffle = shuffle
        self._num_workers = num_workers

    @property
    def path_A(self) -> Path:
        return self._path_A

    @property
    def path_B(self) -> Path:
        return self._path_B

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def image_augmentations(self) -> List[Callable]:
        return self._image_augmentations

    @property
    def data_transforms(self) -> Optional[transforms.Compose]:
        return self._data_transforms

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @property
    def num_workers(self) -> int:
        return self._num_workers
