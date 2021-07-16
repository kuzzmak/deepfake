import PyQt5.QtCore as qtc

from utils import get_file_paths_from_dir


class FaceExtractionWorker(qtc.QObject):

    new_images = qtc.pyqtSignal(list)

    @qtc.pyqtSlot(str)
    def process_faces_folder(self, faces_folder_path: str):
        image_paths = get_file_paths_from_dir(faces_folder_path)
        print('image paths')
        print(image_paths)
        print()
        self.new_images.emit(image_paths)
