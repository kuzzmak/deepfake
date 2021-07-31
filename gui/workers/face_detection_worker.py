from gui.workers.worker import Worker

from message.message import Message

from utils import get_file_paths_from_dir


class FaceDetectionWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):

        faces_directory = msg.body.get_data()
        picture_paths = get_file_paths_from_dir(faces_directory)

        

        print('hello from face detection worker: ')
        print(msg)
        print(picture_paths)
