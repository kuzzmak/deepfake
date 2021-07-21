import abc
from dataclasses import dataclass
from enum import Enum

from enums import CONSOLE_MESSAGE_TYPE, IO_OP_TYPE


class MESSAGE_TYPE(Enum):
    REQUEST = 'request'
    ANSWER = 'answer'


class JOB_TYPE(Enum):
    IO_OPERATION = 'io_operation'
    CONSOLE_PRINT = 'console_print'
    NO_JOB = 'no_job'


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

    # def get_data(self):
    #     if self.type == MESSAGE_TYPE.REQUEST:
    #         if self.body.job_type == JOB_TYPE.CONSOLE_PRINT:
    #             msg_type, message = self.body.get_data()
    #             print('mes type: ', msg_type)
    #             print('messa: ', message)
