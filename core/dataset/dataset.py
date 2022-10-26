from pathlib import Path
from typing import List, Optional, Union

from torch.utils.data import Dataset
from torchvision import transforms as T


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
