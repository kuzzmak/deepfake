from functools import partial
from dataclasses import dataclass

from enums import DIALOG_MESSAGE_ICON, DIALOG_MESSAGE_TYPE, IO_OP_TYPE


@dataclass
class DialogMessage:
    """Class describing one type of a message which can be displayed
    in dialog. Every message has type, icon and text message.

    Returns
    -------
    DialogMessage
        message displayable in dialog
    """
    _message_type: DIALOG_MESSAGE_TYPE
    _message_icon: DIALOG_MESSAGE_ICON
    message: str

    @property
    def message_icon(self):
        return self._message_icon.value

    @property
    def message_type(self):
        return self._message_type.value


class DialogMessages:
    """Common dialog messages where only message text should
    be passed as an argument.
    """

    DELETE = partial(DialogMessage,
                     DIALOG_MESSAGE_TYPE.DELETE,
                     DIALOG_MESSAGE_ICON.DELETE)

    RENAME = partial(DialogMessage,
                     DIALOG_MESSAGE_TYPE.RENAME,
                     DIALOG_MESSAGE_ICON.RENAME)

    WRNING = partial(DialogMessage,
                     DIALOG_MESSAGE_TYPE.WARNING,
                     DIALOG_MESSAGE_ICON.WARNING)


@dataclass
class IO_OP:
    """Describes one IO operation which has a type and
    path to a file on which this operation should be executed.
    """
    type: IO_OP_TYPE
    value: str
