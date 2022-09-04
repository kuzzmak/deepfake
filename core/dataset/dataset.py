import random
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from variables import IMAGE_EXTS


class BaseDataset(Dataset):
    """Simple abstact base class for faceswapping datasets.

    Parameters
    ----------
    dataset_root : Union[str, Path]
        path to the directory with data
    transforms : Optional[T.Compose], optional
        transformations done on every image before sending to the model,
            by default None
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        transforms: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()

        self._dataset_root = Path(dataset_root)
        self._data_paths = self._get_data_paths()
        self._transforms = transforms
        if transforms is None:
            self._transforms = T.Compose([T.ToTensor()])

    def _get_data_paths(self) -> List[Path]:
        """Implementation of this function should fetch all data paths.

        Returns
        -------
        List[Path]
            list of all data paths

        Raises
        ------
        NotImplementedError
            throws if not implemented
        """
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._data_paths)


class FSDataset(BaseDataset):
    """Simple dataset wrapper for the `FS` model. 

    Parameters
    ----------
    dataset_root : Union[str, Path]
        path to the directory with data
    transforms : Optional[T.Compose], optional
        transformations done on every image before sending to the model,
            by default None
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        transforms: Optional[T.Compose] = None,
    ) -> None:
        super().__init__(dataset_root, transforms)

    def _get_data_paths(self) -> List[Path]:
        paths = []
        for p in list(self._dataset_root.glob('*')):
            im_paths = list(p.glob('*'))
            im_paths = list(
                filter(lambda ip: ip.suffix[1:] in IMAGE_EXTS, im_paths)
            )
            paths.append(im_paths)
        return paths

    def __getitem__(self, index: int):
        paths = self._data_paths[index]
        p1 = random.choice(paths)
        p2 = random.choice(paths)
        im1 = self._transforms(Image.open(p1))
        im2 = self._transforms(Image.open(p2))
        return im1, im2
