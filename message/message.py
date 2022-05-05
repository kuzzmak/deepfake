from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from enums import (
    FILE_TYPE,
    IO_OPERATION_TYPE,
    JOB_NAME,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
    BODY_KEY,
)


@dataclass
class Body:

    job_type: JOB_TYPE = JOB_TYPE.NO_JOB
    data: Optional[dict] = field(default_factory=dict)
    finished: Optional[bool] = False


@dataclass
class Message:

    type: MESSAGE_TYPE
    status: MESSAGE_STATUS
    sender: SIGNAL_OWNER
    recipient: SIGNAL_OWNER
    body: Optional[Body] = Body()


class Messages:

    def CONFIGURE_WIDGET(
        sender: SIGNAL_OWNER,
        widget: WIDGET,
        method: str,
        args: List,
        job_name: JOB_NAME,
    ):
        return Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            sender,
            SIGNAL_OWNER.CONFIGURE_WIDGET,
            Body(
                JOB_TYPE.WIDGET_CONFIGURATION,
                {
                    BODY_KEY.WIDGET: widget,
                    BODY_KEY.METHOD: method,
                    BODY_KEY.ARGS: args,
                    BODY_KEY.JOB_NAME: job_name.value,
                }
            )
        )

    def JOB_EXIT():
        return Message(
            MESSAGE_TYPE.JOB_EXIT,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.NO_OWNER,
            SIGNAL_OWNER.JOB_PROGRESS,
        )


class IOOperationBody(Body):

    def __init__(self,
                 io_operation_type: IO_OPERATION_TYPE,
                 file_path: str,
                 new_file_path: Optional[str] = None,
                 file: Optional[np.ndarray] = None,
                 file_type: Optional[FILE_TYPE] = None,
                 resize: Optional[bool] = False,
                 new_size: Optional[int] = None,
                 multipart: Optional[bool] = False,
                 part: Optional[int] = None,
                 total: Optional[int] = None,
                 ):
        super().__init__(
            JOB_TYPE.IO_OPERATION,
            {
                BODY_KEY.IO_OPERATION_TYPE: io_operation_type,
                BODY_KEY.FILE_PATH: file_path,
                BODY_KEY.NEW_FILE_PATH: new_file_path,
                BODY_KEY.FILE: file,
                BODY_KEY.FILE_TYPE: file_type,
                BODY_KEY.RESIZE: resize,
                BODY_KEY.NEW_SIZE: new_size,
                BODY_KEY.MULTIPART: multipart,
                BODY_KEY.PART: part,
                BODY_KEY.TOTAL: total,
            }
        )
