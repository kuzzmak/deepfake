import PyQt5.QtWidgets as qwt

from gui.widgets.picture_viewer import PictureViewer
from gui.widgets.video_player import VideoPlayer

from utils import get_file_paths_from_dir


class DataSelector(qwt.QWidget):

    def __init__(self, data_type: str):
        super().__init__()

        self.data_type = data_type

        self.init_ui()

    def init_ui(self):

        layout = qwt.QVBoxLayout()

        layout.addWidget(qwt.QLabel(text=f'Select {self.data_type} data'))

        button_wgt = qwt.QWidget()
        button_layout = qwt.QHBoxLayout()
        button_wgt.setLayout(button_layout)
        select_video_btn = qwt.QPushButton(text='Select video')
        select_video_btn.clicked.connect(self.select_video)
        select_pictures_btn = qwt.QPushButton(text='Select pictures')
        select_pictures_btn.clicked.connect(self.select_pictures)
        button_layout.addWidget(select_video_btn)
        button_layout.addWidget(select_pictures_btn)

        self.preview_widget = qwt.QStackedWidget()

        self.video_player = VideoPlayer()
        self.preview_widget.addWidget(self.video_player)

        self.picture_viewer = PictureViewer()
        self.preview_widget.addWidget(self.picture_viewer)

        layout.addWidget(button_wgt)
        layout.addWidget(self.preview_widget)

        self.setLayout(layout)

    def select_video(self):
        video_path = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\videos\\interview_woman.mp4"
        self.video_player.video_selection.emit(video_path)
        self.preview_widget.setCurrentWidget(self.video_player)

    def select_pictures(self):
        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\dummy_pics"
        image_paths = get_file_paths_from_dir(directory)
        self.picture_viewer.pictures_added_sig.emit(image_paths)
        self.preview_widget.setCurrentWidget(self.picture_viewer)
