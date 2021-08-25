import os
import pickle

from core.exception import FileDoesNotExistsError, NotDirectoryError
from core.face import Face


class Serializer:

    @staticmethod
    def save_face(face: Face, face_path: str):
        """Saves `Face` object to the directory `face_path` as a pickle file.

        Parameters
        ----------
        face : Face
            face object being saved
        face_path : str
            directory where file should be saved

        Raises
        ------
        NotDirectoryError
            if `face_path` is not a directory
        """
        if not os.path.exists(face_path):
            os.makedirs(face_path)

        if not os.path.isdir(face_path):
            raise NotDirectoryError(face_path)

        # image name, no extension
        image_name = face.raw_image.name.split('.')[0]

        image_path = os.path.join(
            face_path,
            'metadata_' + image_name + '.p',
        )
        pickle.dump(face, open(image_path, 'wb'))

    @staticmethod
    def load_face(face_path: str) -> Face:
        """Load face from the metadata file.

        Parameters
        ----------
        face_path : str
            path of the face metadata

        Returns
        -------
        Face
            face object

        Raises
        ------
        FileDoesNotExistsError
            if `face_path` does not exist
        """
        if not os.path.exists(face_path):
            raise FileDoesNotExistsError(face_path)

        face = pickle.load(open(face_path, "rb"))

        return face
