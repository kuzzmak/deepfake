from dataclasses import dataclass, field
from typing import List, Optional

from enums import (
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
                },
            )
        )

    def JOB_EXIT():
        return Message(
            MESSAGE_TYPE.JOB_EXIT,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.NO_OWNER,
            SIGNAL_OWNER.JOB_PROGRESS,
        )
