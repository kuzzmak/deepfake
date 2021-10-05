from dataclasses import dataclass

from common_structures import CommObject


@dataclass
class PreviewConfiguration:
    """Basic configuration onject for preview of the training process.

    Parameters
    ----------
    show_preview : bool
        should preview be refreshe/showed in gui, by default False
    comm_object : CommObject
        object containing signal for the communication between threads and gui,
            by default None
    """
    show_preview: bool = False
    comm_object: CommObject = None
