from enums import IMAGE_FORMAT


class DeepfakeError(Exception):
    """Base class for exceptions in this application."""

    def __init__(self, message: str):
        self._message = message

    def __str__(self) -> str:
        return self._message


class NoBoundingBoxError(DeepfakeError):
    """Error when Face object doesn't contain bounding box. Face detection
    should be run and the resulting bounding box should be set on the face.
    """

    def __init__(self) -> None:
        super().__init__('No bounding box, face detection not run yet?')


class NoLandmarksError(DeepfakeError):
    """When no face landmark detection was done, this error should be raised.
    """

    def __init__(self):
        super().__init__('No face landmarks, landmark detection not run yet?')


class FileDoesNotExistsError(DeepfakeError):
    """When trying to load file that doesn't exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f'File: {path} doesn\'t exist.')


class UnsupportedImageTypeError(DeepfakeError):
    """Thrown when there is an attempt to load image type other than jpg or
    png."""

    def __init__(self, ext: str) -> None:
        super().__init__(
            f'Tried to load unsupported format: {ext}. Only supported ' +
            f'formats are: {[f.value for f in IMAGE_FORMAT]}.',
        )


class NotFileError(DeepfakeError):
    """Tried to load something that is not a file."""

    def __init__(self, path: str):
        super().__init__(f'Provided path: {path} is not a file.')


class NotDirectoryError(DeepfakeError):
    """Tried to save or load something from something that is not a directory.
    """

    def __init__(self, path: str):
        super().__init__(f'Provided path: {path} is not a directory.')
