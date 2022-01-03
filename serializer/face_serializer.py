import gzip
import os
import pickle

from core.face import Face
from core.exception import FileDoesNotExistsError, NotDirectoryError

from serializer.serializer import Serializer
from utils import construct_file_path


class FaceSerializer(Serializer):
    """Serializer for the `Face` object."""

    def load(path: str) -> Face:
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
        if not os.path.exists(path):
            raise FileDoesNotExistsError(path)

        face = pickle.load(gzip.open(path, "rb"))

        return face

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

        face_path = os.path.join(
            path,
            'metadata_' + face_name + '.p',
        )
        face_path = construct_file_path(face_path)
        obj.path = face_path
        pickle.dump(obj, gzip.open(face_path, 'wb'))
