from dataclasses import dataclass
from typing import Optional

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from core.dataset.dataset import DeepfakeDataset


@dataclass
class DatasetConfiguration:

    faces_path: str
    batch_size: int
    input_shape: int
    data_transforms: Optional[transforms.Compose] = None
    shuffle: bool = True
    num_workers: int = 2

    def __post_init__(self):
        dataset = DeepfakeDataset(
            self.faces_path,
            self.input_shape,
            self.data_transforms,
        )
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
