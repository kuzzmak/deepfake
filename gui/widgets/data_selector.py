import os

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from common_structures import DIRECTORY_NOT_SELECTED_MESSAGE

from enums import CONSOLE_MESSAGE_TYPE, MESSAGE_TYPE, SIGNAL_OWNER

from gui.widgets.base_widget import BaseWidget
from gui.widgets.picture_viewer import PictureViewer
from gui.widgets.video_player import VideoPlayer

from message.message import ConsolePrintMessageBody, Message

from utils import get_file_paths_from_dir


class DataSelector(BaseWidget):

    selected_video = qtc.pyqtSignal(str)
    selected_pictures_directory = qtc.pyqtSignal(str)

    def __init__(self, data_type: str):
        super().__init__()

        self.data_type = data_type

        self.init_ui()

    def init_ui(self):

        self.main_layout = qwt.QVBoxLayout()

        self.main_layout.addWidget(qwt.QLabel(
            text=f'Select {self.data_type} data'))

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
        self.preview_widget.addWidget(qwt.QWidget())

        self.video_player_wgt = qwt.QWidget()
        video_player_wgt_layout = qwt.QVBoxLayout()
        self.video_player_wgt.setLayout(video_player_wgt_layout)

        self.video_player = VideoPlayer()
        video_player_wgt_layout.addWidget(self.video_player)
        self.preview_widget.addWidget(self.video_player_wgt)

        self.picture_viewer = PictureViewer()
        self.preview_widget.addWidget(self.picture_viewer)

        self.preview_label = qwt.QLabel()

        self.frame_extraction_gb = qwt.QGroupBox()
        self.frame_extraction_gb.setTitle('Destination directory for extracted frames')
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Minimum, qwt.QSizePolicy.Maximum)
        self.frame_extraction_gb.setSizePolicy(size_policy)

        box_group_layout = qwt.QVBoxLayout()
        self.frame_extraction_gb.setLayout(box_group_layout)

        left_part_wgt = qwt.QWidget()
        left_part_layout = qwt.QVBoxLayout()
        left_part_wgt.setLayout(left_part_layout)
        select_frames_directory_btn = qwt.QPushButton(text='Select')
        left_part_layout.addWidget(select_frames_directory_btn)

        right_part_wgt = qwt.QWidget()
        right_part_layout = qwt.QHBoxLayout()
        right_part_wgt.setLayout(right_part_layout)
        resize_frames_chk = qwt.QCheckBox(text='Resize frames')
        resize_frames_chk.stateChanged.connect(self.resize_frames_chk_changed)
        right_part_layout.addWidget(resize_frames_chk)
        self.biggest_frame_dim_input = qwt.QLineEdit()
        self.biggest_frame_dim_input.setEnabled(False)
        right_part_layout.addWidget(self.biggest_frame_dim_input)

        row = qwt.QWidget()
        row_layout = qwt.QHBoxLayout()
        row.setLayout(row_layout)
        row_layout.addWidget(left_part_wgt, 0, qtc.Qt.AlignTop)
        row_layout.addWidget(right_part_wgt, 0, qtc.Qt.AlignTop)
        box_group_layout.addWidget(row)

        extract_frames_btn = qwt.QPushButton(text='Extract frames')
        box_group_layout.addWidget(extract_frames_btn)

        video_player_wgt_layout.addWidget(self.frame_extraction_gb)

        self.main_layout.addWidget(button_wgt)
        self.main_layout.addWidget(self.preview_label)
        self.main_layout.addWidget(self.preview_widget)

        self.setLayout(self.main_layout)

    def select_video(self):
        """Select video from which individual frames would be extracted
        and then these frames will be used for face extraction process.
        """
        video_path = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\videos\\interview_woman.mp4"
        # options = qwt.QFileDialog.Options()
        # options |= qwt.QFileDialog.DontUseNativeDialog
        # video_path, _ = qwt.QFileDialog.getOpenFileName(
        #     self,
        #     'Select video file',
        #     "data/videos",
        #     "Video files (*.mp4)",
        #     options=options)

        if video_path:
            msg = Message(
                MESSAGE_TYPE.REQUEST,
                ConsolePrintMessageBody(
                    CONSOLE_MESSAGE_TYPE.LOG,
                    f'{self.data_type} video selected from: {video_path}'
                )
            )

            self.video_player.video_selection.emit(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.preview_label.setText(
                f'Preview of the: {video_name}')
            self.preview_widget.setCurrentWidget(self.video_player_wgt)

        else:
            msg = DIRECTORY_NOT_SELECTED_MESSAGE

        self.signals[SIGNAL_OWNER.CONOSLE].emit(msg)

    def select_pictures(self):
        """Select directory with faces which would be used for face
        extraction process.
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self,
        #     "getExistingDirectory",
        #     "./"
        # )
        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\dummy_pics"

        if directory:
            self.data_directory = directory
            self.preview_widget.setCurrentWidget(self.picture_viewer)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                msg = Message(
                    MESSAGE_TYPE.REQUEST,
                    ConsolePrintMessageBody(
                        CONSOLE_MESSAGE_TYPE.WARNING,
                        f'No images were found in: {directory}.'
                    )
                )

            else:
                self.picture_viewer.pictures_added_sig.emit(image_paths)

                msg = Message(
                    MESSAGE_TYPE.REQUEST,
                    ConsolePrintMessageBody(
                        CONSOLE_MESSAGE_TYPE.INFO,
                        'Loaded: {} images from: {}.'.format(
                            len(image_paths),
                            directory
                        )
                    )
                )

                self.preview_label.setText(
                    f'Preview of pictures in {directory} directory.')

        else:

            msg = DIRECTORY_NOT_SELECTED_MESSAGE

        self.signals[SIGNAL_OWNER.CONOSLE].emit(msg)

    @qtc.pyqtSlot(int)
    def resize_frames_chk_changed(self, value: int):
        """Enables widget for biggest frame dimension input when is checked.

        Parameters
        ----------
        value : int
            new checkbox value
        """
        if value == 2:  # checkbox checked value
            self.biggest_frame_dim_input.setEnabled(True)
        else:
            self.biggest_frame_dim_input.setEnabled(False)
