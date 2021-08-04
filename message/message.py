from dataclasses import dataclass, field
from typing import NewType, Optional

import numpy as np

from enums import (
    CONSOLE_MESSAGE_TYPE,
    FACE_DETECTION_ALGORITHM,
    FILE_TYPE,
    IO_OPERATION_TYPE,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    WIDGET,
    BODY_KEY,
)


# class ConsolePrintMessageBody(MessageBody):

#     def __init__(self, type: CONSOLE_MESSAGE_TYPE, message: str):
#         super().__init__(JOB_TYPE.CONSOLE_PRINT)

#         self.type = type
#         self.message = message

#     def get_data(self):
#         return self.type, self.message


# class FrameExtractionMessageBody(MessageBody):

#     def __init__(self,
#                  video_path: str,
#                  destination_directory: str,
#                  resize: bool,
#                  new_size: int,
#                  image_format: str):
#         super().__init__(JOB_TYPE.FRAME_EXTRACTION)

#         self.video_path = video_path
#         self.destination_directory = destination_directory
#         self.resize = resize
#         self.new_size = new_size
#         self.image_format = image_format

#     def get_data(self):
#         return self.video_path, self.destination_directory, self.resize, self.new_size, self.image_format


# class IO_OperationMessageBody(MessageBody):

#     def __init__(self,
#                  type: IO_OP_TYPE,
#                  file_path: str,
#                  new_file_path: Optional[str] = '',
#                  file: Optional[np.ndarray] = None,
#                  resize: bool = False,
#                  new_size: int = None,
#                  multipart: Optional[bool] = False,
#                  part: Optional[int] = None,
#                  total: Optional[int] = None):
#         super().__init__(JOB_TYPE.IO_OPERATION)

#         self.type = type
#         self.file_path = file_path
#         self.new_file_path = new_file_path
#         self.file = file
#         self.resize = resize
#         self.new_size = new_size
#         self.multipart = multipart
#         self.part = part
#         self.total = total

#     def get_data(self):
#         return (
#             self.type,
#             self.file_path,
#             self.new_file_path,
#             self.file,
#             self.resize,
#             self.new_size,
#             self.multipart,
#             self.part,
#             self.total,
#         )


# class ConfigureWidgetMessageBody(MessageBody):

#     def __init__(self,
#                  widget: WIDGET,
#                  widget_method: str,
#                  method_args: List):
#         super().__init__(JOB_TYPE.WIDGET_CONFIGURATION)

#         self.widget = widget
#         self.widget_method = widget_method
#         self.method_args = method_args

#     def get_data(self):
#         return self.widget, self.widget_method, self.method_args


# class FaceDetectionMessageBody(MessageBody):

#     def __init__(self,
#                  faces_directory: str,
#                  model_path: str,
#                  algorithm: Optional[FACE_DETECTION_ALGORITHM]
#                  = FACE_DETECTION_ALGORITHM.S3FD):
#         super().__init__(JOB_TYPE.FACE_DETECTION)

#         self.faces_directory = faces_directory
#         self.model_path = model_path
#         self.algorithm = algorithm

#     def get_data(self):
#         return self.faces_directory, self.model_path, self.algorithm


# class AnswerBody(MessageBody):

#     def __init__(self, status: MESSAGE_STATUS, finished: bool):
#         super().__init__(JOB_TYPE.IO_OPERATION)

#         self.status = status
#         self.finished = finished

#     def get_data(self):
#         return self.status, self.finished

@dataclass
class Body:

    job_type: JOB_TYPE = JOB_TYPE.NO_JOB
    data: Optional[dict] = field(default_factory=dict)
    finished: Optional[bool] = False


@dataclass
class Message:

    type: MESSAGE_TYPE
    status: MESSAGE_STATUS
    body: Optional[Body] = Body()


class Messages:

    def CONSOLE_PRINT(msg_type: CONSOLE_MESSAGE_TYPE, message: str):
        return Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            Body(
                JOB_TYPE.CONSOLE_PRINT,
                {
                    'console_message_type': msg_type,
                    'message': message
                }
            )
        )

    DIRECTORY_NOT_SELECTED = CONSOLE_PRINT(
        CONSOLE_MESSAGE_TYPE.WARNING,
        'No directory selected.'
    )

    NO_IMAGES_FOUND = CONSOLE_PRINT(
        CONSOLE_MESSAGE_TYPE.WARNING,
        'No images found in directory.'
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
        super().__init__(JOB_TYPE.IO_OPERATION,
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
