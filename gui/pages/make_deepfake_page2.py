from gui.widgets.video_player import VideoPlayer
from gui.widgets.picture_viewer import PictureViewer
from names import MAKE_DEEPFAKE_PAGE_TITLE, START_PAGE_NAME
from gui.pages.page import CONSOLE_MESSAGE_TYPE, Page
from gui.templates.make_deepfake_page_2 import Ui_make_deepfake_page
import PyQt5.QtWidgets as qwt
import os
from utils import get_file_paths_from_dir
import PyQt5.QtCore as qtc
import time


class Worker(qtc.QObject):

    incremented_val = qtc.pyqtSignal(int)

    @qtc.pyqtSlot(int)
    def increment_value(self, value: int):
        new_value = value
        while new_value < 100:
            new_value += 1
            time.sleep(0.05)
            self.incremented_val.emit(new_value)



class MakeDeepfakePage2(Page, Ui_make_deepfake_page):

    new_val_requested = qtc.pyqtSignal(int)

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name='make_deepfake_page_2', *args, **kwargs)

        self.setupUi(self)

        self.picture_viewer = PictureViewer()
        self.preview_widget.addWidget(self.picture_viewer)

        self.video_player = VideoPlayer()
        self.preview_widget.addWidget(self.video_player)

        cpus_num = os.cpu_count()
        init_cpus_num = cpus_num // 2
        self.number_of_threads_slider.setMaximum(cpus_num)
        self.number_of_threads_slider.setSliderPosition(init_cpus_num)
        self.number_of_threads_label.setText(str(init_cpus_num))
        self.number_of_threads_slider.sliderMoved.connect(self.slider_moved)

        # self.enable_detection_algorithm_tab(False)

        # self.face_extraction_progress.hide()
        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.select_pictures_btn.clicked.connect(self.select_pictures)
        self.select_video_btn.clicked.connect(self.select_video)
        self.start_detection_btn.clicked.connect(self.start_detection)

        self.worker = Worker()
        self.worker_thread = qtc.QThread()
        self.worker.incremented_val.connect(self.increment_progress)
        self.new_val_requested.connect(self.worker.increment_value)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()


        self.face_extraction_progress.valueChanged.connect(self.progress_value_changed)

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
        self.new_val_requested.emit(0)
        
        # self.face_extraction_progress.show()

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
        directory = qwt.QFileDialog.getExistingDirectory(
            self, "getExistingDirectory", "./")
        if directory:
            self.preview_widget.setCurrentWidget(self.picture_viewer)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                self.print(
                    f'No images were found in: {directory}.', CONSOLE_MESSAGE_TYPE.WARNING)
            else:
                for img_path in image_paths:
                    self.picture_viewer.picture_added_sig.emit(img_path)

                message = 'Loaded: {} images from: {}.'.format(
                    len(image_paths), directory)
                self.print(message, CONSOLE_MESSAGE_TYPE.LOG)
                self.set_preview_label_text(
                    'Preview of pictures in: ' + directory + ' folder.')
        else:
            self.print('No picture folder was selected.',
                       CONSOLE_MESSAGE_TYPE.WARNING)
