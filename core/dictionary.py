from __future__ import annotations
from _collections_abc import dict_keys, dict_values
import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


def _map_to_numpy_array(data: Dict[str, list]) -> Dict[str, np.ndarray]:
    return dict(zip(data, map(lambda v: np.array(v), data.values())))


def _map_to_list(data: Dict[str, np.ndarray]) -> Dict[str, list]:
    return dict(zip(data, map(lambda v: v.tolist(), data.values())))


class Dictionary:
    """Simple structure for storing key value pairs where key is of a type str
    and value can be either list of numpy.ndarray.

    Args:
        data (Dict[str, Union[list, np.ndarray]]): dictionary of values,
            by default dict
    """

    def __init__(
        self,
        data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if data is not None:
            self._data = data
        else:
            self._data = dict()

    def add(self, key: str, value: np.ndarray) -> None:
        """Adds key value pair.

        Args:
            key (str): key
            value (np.ndarray): value
        """
        self._data[key] = value

    def keys(self) -> dict_keys[str, np.ndarray]:
        return self._data.keys()

    def values(self) -> dict_values[str, np.ndarray]:
        return self._data.values()

    def __getitem__(self, key: str):
        return self._data.get(key, None)

    def remove(self, key: str) -> None:
        """Removes value on key.

        Args:
            key (str): key of the value
        """
        self._data.pop(key, None)

    def __len__(self) -> int:
        """Returns the size of the dictionary, i.e. number of keys.

        Returns:
            int: size of the dictionary
        """
        return len(self._data)

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
            json.dump(_map_to_list(self._data), f)

    @staticmethod
    def load(path: Union[Path, str]) -> Dictionary:
        """Loads dictionary from some file.

        Args:
            path (Union[Path, str]): path to the file

        Returns:
            Dictionary: constructed dictionary
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return Dictionary(_map_to_numpy_array(data))
