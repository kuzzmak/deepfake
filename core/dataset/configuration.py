from dataclasses import dataclass
from typing import Optional

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
        size of the square image on the input
    data_transforms : Optional[transforms.Compose], optional
        transformations for the dataset, by default None
    load_into_memory : bool
        should all dataser be loaded into RAM or GPU?, by default False
    shuffle : bool
        should the dataset be shuffled, by default True
    num_workers : int
        number of threads used for dataset loading, by default 2
    """
    metadata_path_A: str
    metadata_path_B: str
    batch_size: int
    input_shape: int
    data_transforms: Optional[transforms.Compose] = None
    load_into_memory: bool = False
    shuffle: bool = True
    num_workers: int = 2
