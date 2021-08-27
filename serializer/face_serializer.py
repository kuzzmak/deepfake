import os
import pickle

from core.face import Face
from core.exception import FileDoesNotExistsError, NotDirectoryError

from serializer.serializer import Serializer


class FaceSerializer(Serializer):

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

        face = pickle.load(open(path, "rb"))

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
        image_name = obj.raw_image.name.split('.')[0]

        image_path = os.path.join(
            path,
            'metadata_' + image_name + '.p',
        )
        pickle.dump(obj, open(image_path, 'wb'))
