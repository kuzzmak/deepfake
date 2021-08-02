import abc
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np

from enums import (
    CONSOLE_MESSAGE_TYPE,
    FACE_DETECTION_ALGORITHM,
    IO_OP_TYPE,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    WIDGET
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

    def __init__(self,
                 video_path: str,
                 destination_directory: str,
                 resize: bool,
                 new_size: int,
                 image_format: str):
        super().__init__(JOB_TYPE.FRAME_EXTRACTION)

        self.video_path = video_path
        self.destination_directory = destination_directory
        self.resize = resize
        self.new_size = new_size
        self.image_format = image_format

    def get_data(self):
        return self.video_path, self.destination_directory, self.resize, self.new_size, self.image_format


class IO_OperationMessageBody(MessageBody):

    def __init__(self,
                 type: IO_OP_TYPE,
                 file_path: str,
                 new_file_path: Optional[str] = '',
                 file: Optional[np.ndarray] = None,
                 resize: bool = False,
                 new_size: int = None,
                 multipart: Optional[bool] = False,
                 part: Optional[int] = None,
                 total: Optional[int] = None):
        super().__init__(JOB_TYPE.IO_OPERATION)

        self.type = type
        self.file_path = file_path
        self.new_file_path = new_file_path
        self.file = file
        self.resize = resize
        self.new_size = new_size
        self.multipart = multipart
        self.part = part
        self.total = total

    def get_data(self):
        return (
            self.type,
            self.file_path,
            self.new_file_path,
            self.file,
            self.resize,
            self.new_size,
            self.multipart,
            self.part,
            self.total,
        )


class ConfigureWidgetMessageBody(MessageBody):

    def __init__(self,
                 widget: WIDGET,
                 widget_method: str,
                 method_args: List):
        super().__init__(JOB_TYPE.WIDGET_CONFIGURATION)

        self.widget = widget
        self.widget_method = widget_method
        self.method_args = method_args

    def get_data(self):
        return self.widget, self.widget_method, self.method_args


class FaceDetectionMessageBody(MessageBody):

    def __init__(self,
                 faces_directory: str,
                 model_path: str,
                 algorithm: Optional[FACE_DETECTION_ALGORITHM]
                 = FACE_DETECTION_ALGORITHM.S3FD):
        super().__init__(JOB_TYPE.FACE_DETECTION)

        self.faces_directory = faces_directory
        self.model_path = model_path
        self.algorithm = algorithm

    def get_data(self):
        return self.faces_directory, self.model_path, self.algorithm


class AnswerBody(MessageBody):

    def __init__(self, status: MESSAGE_STATUS, finished: bool):
        super().__init__(JOB_TYPE.IO_OPERATION)

        self.status = status
        self.finished = finished

    def get_data(self):
        return self.status, self.finished


@dataclass
class RequestBody(MessageBody):

    job_type: JOB_TYPE
    data: dict = field(default_factory=dict)

    def get_data(self):
        return super().get_data()


@dataclass
class AnswerBody2(MessageBody):

    status: MESSAGE_STATUS
    job_type: Optional[JOB_TYPE] = JOB_TYPE.NO_JOB
    finished: Optional[bool] = False
    data: Optional[dict] = field(default_factory=dict)

    def get_data(self):
        return super().get_data()


@dataclass
class Message:
    type: MESSAGE_TYPE
    body: MessageBody
