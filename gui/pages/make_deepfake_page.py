import os
from typing import List

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import CONSOLE_MESSAGE_TYPE, Page
from gui.templates.make_deepfake_page import Ui_make_deepfake_page
from gui.widgets.video_player import VideoPlayer
from gui.widgets.picture_viewer import PictureViewer
from gui.workers.face_extraction_worker import FaceExtractionWorker
from gui.workers.worker import Worker

from utils import get_file_paths_from_dir

from names import MAKE_DEEPFAKE_PAGE_NAME, MAKE_DEEPFAKE_PAGE_TITLE

from resources.icons import icons


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    new_val_requested = qtc.pyqtSignal(int)
    faces_extraction_requested = qtc.pyqtSignal(str)

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name=MAKE_DEEPFAKE_PAGE_NAME, *args, **kwargs)

        self.setupUi(self)

        self.picture_viewer_tab_1 = PictureViewer(self.app)
        self.preview_widget.addWidget(self.picture_viewer_tab_1)

        self.video_player = VideoPlayer()
        self.preview_widget.addWidget(self.video_player)

        cpus_num = os.cpu_count()
        init_cpus_num = cpus_num // 2
        self.number_of_threads_slider.setMaximum(cpus_num)
        self.number_of_threads_slider.setSliderPosition(init_cpus_num)
        self.number_of_threads_label.setText(str(init_cpus_num))
        self.number_of_threads_slider.sliderMoved.connect(self.slider_moved)

        # self.enable_detection_algorithm_tab(False)

        self.face_extraction_progress.hide()
        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.select_pictures_btn.clicked.connect(self.select_pictures)
        self.select_video_btn.clicked.connect(self.select_video)
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.start_detection_btn.setIcon(qtg.QIcon(qtg.QPixmap(':/play.svg')))
        self.enable_widget(self.start_detection_btn, False)
        self.select_faces_folder_btn.clicked.connect(self.select_faces_folder)

        self.picture_viewer_tab_2 = PictureViewer(self.app)
        self.image_viewer_layout.addWidget(self.picture_viewer_tab_2)

        self.worker = Worker()
        self.worker_thread = qtc.QThread()
        self.worker.incremented_val.connect(self.increment_progress)
        self.new_val_requested.connect(self.worker.increment_value)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self.face_extraction_worker = FaceExtractionWorker()
        self.face_extraction_worker_thread = qtc.QThread()
        self.face_extraction_worker.new_images.connect(
            self.add_new_faces_to_image_viewer)
        self.faces_extraction_requested.connect(
            self.face_extraction_worker.process_faces_folder)
        self.face_extraction_worker.moveToThread(
            self.face_extraction_worker_thread)
        self.face_extraction_worker_thread.start()

        self.select_pictures()

        self.face_extraction_progress.valueChanged.connect(
            self.progress_value_changed)

    def select_faces_folder(self):
        directory = qwt.QFileDialog.getExistingDirectory(
            self, "getExistingDirectory", "./")
        if directory:
            self.selected_faces_folder_label.setText(directory)
            self.enable_widget(self.start_detection_btn, True)

    def add_new_faces_to_image_viewer(self, image_paths: List[str]):
        print('from fun: ', image_paths)

    def progress_value_changed(self, value: int):
        if value == 100:
            msg = qwt.QMessageBox(self)
            msg.setIcon(qwt.QMessageBox.Information)
            msg.setText("Face extraction successful.")
            msg.setInformativeText("Extracted faces are shown below.")
            msg.setWindowTitle("Face extraction information")
            msg.setStandardButtons(qwt.QMessageBox.Ok)
            msg.exec_()

    def increment_progress(self, new_val: int):
        self.face_extraction_progress.setValue(new_val)

    def start_detection(self):
        self.face_extraction_progress.show()
        self.new_val_requested.emit(0)
        self.faces_extraction_requested.emit(
            'C:/Users/tonkec/Documents/deepfake/dummy_pics')

    def slider_moved(self, position: int):
        self.number_of_threads_label.setText(str(position))

    def set_preview_label_text(self, text: str):
        self.preview_label.setText(text)

    def enable_detection_algorithm_tab(self, enable: bool):
        self.tab_widget.setTabEnabled(1, enable)

    def select_video(self):
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        video_path, _ = qwt.QFileDialog.getOpenFileName(
            self, 'Select video file', "data/videos", "Video files (*.mp4)", options=options)
        if video_path:
            self.video_player.video_selection.emit(video_path)
            self.preview_widget.setCurrentWidget(self.video_player)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            self.set_preview_label_text(
                'Preview of: ' + video_name + ' video.')
            message = 'Loaded video from: {}'.format(video_path)
            self.print(message, CONSOLE_MESSAGE_TYPE.LOG)

            self.enable_detection_algorithm_tab(True)
        else:
            self.print('No video folder was selected.',
                       CONSOLE_MESSAGE_TYPE.WARNING)

    def select_pictures(self):

        directory = 'C:\\Users\\tonkec\\Documents\\deepfake\\dummy_pics'

        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self, "getExistingDirectory", "./")
        if directory:
            self.preview_widget.setCurrentWidget(self.picture_viewer_tab_1)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                self.print(
                    f'No images were found in: {directory}.', CONSOLE_MESSAGE_TYPE.WARNING)
            else:
                import numpy as np

                im = np.random.randint(0, 255, [100, 50, 3], np.uint8)

                image_paths.append(im)
                self.picture_viewer_tab_1.pictures_added_sig.emit(image_paths)

                message = 'Loaded: {} images from: {}.'.format(
                    len(image_paths), directory)
                self.print(message, CONSOLE_MESSAGE_TYPE.LOG)
                self.set_preview_label_text(
                    'Preview of pictures in: ' + directory + ' folder.')
        else:
            self.print('No picture folder was selected.',
                       CONSOLE_MESSAGE_TYPE.WARNING)
