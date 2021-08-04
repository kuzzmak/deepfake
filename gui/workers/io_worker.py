import cv2 as cv

from message.message import Message

from gui.workers.worker import Worker

from enums import BODY_KEY, FILE_TYPE, IO_OPERATION_TYPE, MESSAGE_STATUS, MESSAGE_TYPE, SIGNAL_OWNER

from utils import resize_image_retain_aspect_ratio


class IO_Worker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):

        data = msg.body.data
        io_operation_type = data[BODY_KEY.IO_OPERATION_TYPE]
        file_path = data[BODY_KEY.FILE_PATH]
        file_type = data[BODY_KEY.FILE_TYPE]

        if io_operation_type == IO_OPERATION_TYPE.SAVE:

            file = data[BODY_KEY.FILE]

            if file_type == FILE_TYPE.IMAGE:

                resize = data[BODY_KEY.RESIZE]

                if resize:

                    new_size = data[BODY_KEY.NEW_SIZE]

                    file = resize_image_retain_aspect_ratio(file, new_size)

                cv.imwrite(file_path, file)

        multipart = data[BODY_KEY.MULTIPART]
        if multipart:
            part = data[BODY_KEY.PART]
            total = data[BODY_KEY.TOTAL]
            print(str(part) + '/' + str(total))

        # op_type, \
        #     file_path, \
        #     new_file_path, \
        #     file, \
        #     resize, \
        #     max_img_size_per_dim, \
        #     multipart, \
        #     part, \
        #     total = msg.body.get_data()

        ...
        # if op_type == IO_OP_TYPE.SAVE:
        #     if file is not None:

        #         if resize:
        #             file = resize_image_retain_aspect_ratio(
        #                 file, max_img_size_per_dim)

        #         cv.imwrite(file_path, file)
        #         if multipart:
        #             msg = Message(
        #                 MESSAGE_TYPE.ANSWER,
        #                 AnswerBody(
        #                     MESSAGE_STATUS.OK,
        #                     part == total,
        #                 )
        #             )
        #             self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)
        #         else:
        #             ...
