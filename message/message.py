import abc
from dataclasses import dataclass

from enums import (
    CONSOLE_MESSAGE_TYPE,
    IO_OP_TYPE,
    JOB_TYPE,
    MESSAGE_TYPE
)


class MessageBody(metaclass=abc.ABCMeta):

    def __init__(self, job_type: JOB_TYPE = JOB_TYPE.NO_JOB):
        self.job_type = job_type

    @abc.abstractclassmethod
    def get_data(self):
        ...


class ConsolePrintMessageBody(MessageBody):

    def __init__(self, type: CONSOLE_MESSAGE_TYPE, message: str):
        super().__init__(JOB_TYPE.CONSOLE_PRINT)

        self.type = type
        self.message = message

    def get_data(self):
        return self.type, self.message


class FrameExtractionMessageBody(MessageBody):

    def __init__(self, video_path: str,
                 destination_directory: str,
                 image_format: str):
        super().__init__(JOB_TYPE.FRAME_EXTRACTION)

        self.video_path = video_path
        self.destination_directory = destination_directory
        self.image_format = image_format

    def get_data(self):
        return self.video_path, self.destination_directory, self.image_format


class IO_OperationMessageBody(MessageBody):

    def __init__(self, type: IO_OP_TYPE, file_path: str):
        super().__init__(JOB_TYPE.IO_OPERATION)

        self.type = type
        self.file_path = file_path

    def get_data(self):
        return self.type, self.file_path


@dataclass
class Message:
    type: MESSAGE_TYPE
    body: MessageBody
