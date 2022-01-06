from dataclasses import dataclass, field
from typing import Callable, List, Optional

from torchvision import transforms


@dataclass
class DatasetConfiguration:
    """Configuration class for the `Deepfake` dataset.

    Parameters
    ----------
    metadata_path_A : str
        path of the directory containing `Face` metadata of person A
    metadata_path_B : str
        path of the directory containing `Face` metadata of person B
    batch_size : int
        batch size
    input_shape : int
        size of the square image on the model input
    output_shape : int
        size of the square image on the model output
    image_augmentaions : List[Callable]
        list of functions for image augmentation
    size : int
        size of the dataset, by default -1, load all
    data_transforms : Optional[transforms.Compose], optional
        transformations for the dataset, by default None
    shuffle : bool
        should the dataset be shuffled, by default True
    num_workers : int
        number of threads used for dataset loading, by default 2
    """
    metadata_path_A: str
    metadata_path_B: str
    batch_size: int
    input_shape: int
    output_shape: int
    image_augmentations: List[Callable] = field()
    size: int = -1
    data_transforms: Optional[transforms.Compose] = None
    shuffle: bool = True
    num_workers: int = 2
