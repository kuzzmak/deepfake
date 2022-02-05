import json
from pathlib import Path
from typing import Union

import numpy as np


class Dictionary:
    """Simple structure for storing key value pairs where key is of a type str
    and value can be either list of numpy.ndarray.
    """

    def __init__(self) -> None:
        self._data = dict()

    def add(self, key: str, value: Union[list, np.ndarray]) -> None:
        """Adds key value pair.

        Args:
            key (str): key
            value (Union[list, np.ndarray]): value
        """
        val = value.tolist() if isinstance(value, np.ndarray) else value
        self._data[key] = val

    def save(self, path: Path, file_type: str = 'json') -> None:
        """Saves dictionary to the disk in a form of json file.

        Args:
            path (Path): path to the dictionary
            file_type (str, optional): saves dictionary as a file type.
                Defaults to 'json'.

        Raises:
            NotImplementedError: raises error if some other than json type is
                passed as an argument
        """
        if file_type is not 'json':
            raise NotImplementedError
        with open(path, 'w') as f:
            json.dump(self._data, f)
