from dataclasses import dataclass, field
from typing import Callable, List, Optional

from torchvision import transforms


@dataclass
class DatasetConfiguration:
    """Configuration class for the `Deepfake` dataset.

    Parameters
    ----------
    path_A : str
        path of the directory containing `Face` metadata of person A
    path_B : str
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
    path_A: str
    path_B: str
    input_size: int
    output_size: int
    batch_size: int
    image_augmentations: List[Callable] = field()
    data_transforms: Optional[transforms.Compose] = None
    shuffle: bool = True
    num_workers: int = 2
