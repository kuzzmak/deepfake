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
