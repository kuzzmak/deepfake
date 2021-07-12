from gui.widgets.video_player import VideoPlayer
from gui.widgets.picture_viewer import PictureViewer
from names import MAKE_DEEPFAKE_PAGE_TITLE, START_PAGE_NAME
from gui.pages.page import CONSOLE_MESSAGE_TYPE, Page
from gui.templates.make_deepfake_page_2 import Ui_make_deepfake_page
import PyQt5.QtWidgets as qwt
import os
from utils import get_file_paths_from_dir


class MakeDeepfakePage2(Page, Ui_make_deepfake_page):

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

        self.enable_detection_algorithm_tab(False)

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.select_pictures_btn.clicked.connect(self.select_pictures)
        self.select_video_btn.clicked.connect(self.selecte_video)

    def slider_moved(self, position: int):
        self.number_of_threads_label.setText(str(position))

    def set_preview_label_text(self, text: str):
        self.preview_label.setText(text)

    def enable_detection_algorithm_tab(self, enable: bool):
        self.tab_widget.setTabEnabled(1, enable)

    def selecte_video(self):
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
