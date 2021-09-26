from dataclasses import dataclass
from typing import Optional

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from core.dataset.dataset import DeepfakeDataset
from enums import DEVICE


@dataclass
class DatasetConfiguration:
    """Configuration class for the `Deepfake` dataset.

    Parameters
    ----------
    metadata_path : str
        path of the directory containing `Face` metadata
    batch_size : int
        batch size
    input_shape : int
        size of the square image on the input
    device : DEVICE
        send data to which device, by default DEVICE.CPU
    data_transforms : Optional[transforms.Compose], optional
        transformations for the dataset, by default None
    load_into_memory : bool
        should all dataser be loaded into RAM or GPU?, by default False
    shuffle : bool
        should the dataset be shuffled, by default True
    num_workers : int
        number of threads used for dataset loading, by default 2
    """
    metadata_path: str
    batch_size: int
    input_shape: int
    device: DEVICE = DEVICE.CPU
    data_transforms: Optional[transforms.Compose] = None
    load_into_memory: bool = False
    shuffle: bool = True
    num_workers: int = 2

    def __post_init__(self):
        dataset = DeepfakeDataset(
            metadata_path=self.metadata_path,
            input_shape=self.input_shape,
            load_into_memory=self.load_into_memory,
            device=self.device,
            transforms=self.data_transforms,
        )
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            # num_workers=self.num_workers,
            num_workers=0,  # > 0 not working on Windows
        )
