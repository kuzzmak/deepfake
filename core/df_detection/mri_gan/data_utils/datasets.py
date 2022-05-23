from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class SimpleImageFolder(Dataset):

    def __init__(self, root: Path, transforms=None):
        super().__init__()

        all_files = list(root.glob('*.*'))
        self.data_list = [f.absolute() for f in all_files]
        self.data_len = len(self.data_list)
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_list[index])
        if self.transforms:
            img = self.transforms(img)
        return img

    def __len__(self) -> int:
        return self.data_len
