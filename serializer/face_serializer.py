import gzip
import os
from pathlib import Path
import pickle
from typing import Union

from core.face import Face
from core.exception import FileDoesNotExistsError, NotDirectoryError
from serializer.serializer import Serializer
from utils import construct_file_path


class FaceSerializer(Serializer):
    """Serializer for the `Face` object."""

    def load(path: Union[str, Path]) -> Face:
        """Load face from the metadata file.

        Parameters
        ----------
        path : str
            path of the face metadata

        Returns
        -------
        Face
            face object

        Raises
        ------
        FileDoesNotExistsError
            if `path` does not exist
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileDoesNotExistsError(str(path))
        return pickle.load(gzip.open(path, "rb"))

    def save(obj: Face, path: str):
        """Saves `Face` object to the directory `path` as a pickle file.

        Parameters
        ----------
        obj : Face
            face object being saved
        path : str
            directory where file should be saved

        Raises
        ------
        NotDirectoryError
            if `path` is not a directory
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.isdir(path):
            raise NotDirectoryError(path)

        # image name, no extension
        face_name = obj.raw_image.name.split('.')[0]

        face_path = os.path.join(path, face_name + '.p')
        face_path = construct_file_path(face_path)
        obj.path = face_path
        obj.name = face_path.name
        pickle.dump(obj, gzip.open(face_path, 'wb'))
