from typing import TypeVar


T = TypeVar('T')


class Serializer:
    """Base class which every other `Serializer` class should implement.
    Provides functionality for saving and loading of a specific object.
    """

    @staticmethod
    def load(path: str) -> T:
        """Loads object from the specific path.

        Parameters
        ----------
        path : str
            path to the object

        Returns
        -------
        T
            object

        Raises
        ------
        NotImplementedError
            if `Serializer` class did not yet implement `load` function
        """
        raise NotImplementedError

    @staticmethod
    def save(obj: T, path: str):
        """Saves object `obj` to the specified path `path`.

        Parameters
        ----------
        obj : T
            object for saving
        path : str
            path where object will be saved

        Raises
        ------
        NotImplementedError
            if `Serializer` class did not yet implement `save` function
        """
        raise NotImplementedError
