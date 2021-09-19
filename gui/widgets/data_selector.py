import logging
import os
from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import (
    CONSOLE_MESSAGE_TYPE,
    DATA_TYPE,
    IMAGE_FORMAT,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    BODY_KEY,
)

from gui.widgets.base_widget import BaseWidget
from gui.widgets.picture_viewer import PictureViewer
from gui.widgets.video_player import VideoPlayer

from message.message import (
    Body,
    Message,
    Messages,
)

from utils import get_file_paths_from_dir

logger = logging.getLogger(__name__)


class DataSelector(BaseWidget):

    selected_video = qtc.pyqtSignal(str)
    selected_pictures_directory = qtc.pyqtSignal(str)

    def __init__(self,
                 data_type: DATA_TYPE,
                 signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict()):
        super().__init__(signals)

        self.data_type = data_type
        self.data_directory = None
        self.video_path = None
        self.biggest_frame_dim_value = None

        self.init_ui()

    def init_ui(self):

        self.main_layout = qwt.QVBoxLayout()

        self.main_layout.addWidget(qwt.QLabel(
            text=f'Select {self.data_type.value} data'))

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
        self.frame_extraction_gb.setTitle(
            'Destination directory for extracted frames')
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Minimum, qwt.QSizePolicy.Maximum)
        self.frame_extraction_gb.setSizePolicy(size_policy)

        box_group_layout = qwt.QVBoxLayout()
        self.frame_extraction_gb.setLayout(box_group_layout)

        left_part_wgt = qwt.QWidget()
        left_part_layout = qwt.QVBoxLayout()
        left_part_wgt.setLayout(left_part_layout)
        select_frames_directory_btn = qwt.QPushButton(text='Select')
        select_frames_directory_btn.clicked.connect(
            self.select_frames_directory)
        left_part_layout.addWidget(select_frames_directory_btn)

        right_part_wgt = qwt.QWidget()
        right_part_layout = qwt.QHBoxLayout()
        right_part_wgt.setLayout(right_part_layout)

        self.resize_frames_chk = qwt.QCheckBox(text='Resize frames')
        self.resize_frames_chk.stateChanged.connect(
            self.resize_frames_chk_changed)
        right_part_layout.addWidget(self.resize_frames_chk)

        self.biggest_frame_dim_input = qwt.QLineEdit()
        self.biggest_frame_dim_input.textChanged.connect(
            self.biggest_frame_dim_input_text_changed)
        self.biggest_frame_dim_input.setText(str(640))
        self.enable_widget(self.biggest_frame_dim_input, False)
        right_part_layout.addWidget(self.biggest_frame_dim_input)

        right_part_layout.addWidget(qwt.QLabel(text='Format: '))

        image_format_dropdown = qwt.QComboBox()
        image_format_dropdown.addItem(IMAGE_FORMAT.PNG.value)
        image_format_dropdown.addItem(IMAGE_FORMAT.JPG.value)
        right_part_layout.addWidget(image_format_dropdown)

        row = qwt.QWidget()
        row_layout = qwt.QHBoxLayout()
        row.setLayout(row_layout)
        row_layout.addWidget(left_part_wgt, 0, qtc.Qt.AlignTop)
        row_layout.addWidget(right_part_wgt, 0, qtc.Qt.AlignTop)
        box_group_layout.addWidget(row)

        self.extract_frames_btn = qwt.QPushButton(text='Extract frames')
        self.extract_frames_btn.clicked.connect(self.extract_frames)
        self.enable_widget(self.extract_frames_btn, False)
        box_group_layout.addWidget(self.extract_frames_btn)

        video_player_wgt_layout.addWidget(self.frame_extraction_gb)

        self.main_layout.addWidget(button_wgt)
        self.main_layout.addWidget(self.preview_label)
        self.main_layout.addWidget(self.preview_widget)

        self.setLayout(self.main_layout)

    def select_video(self):
        """Select video from which individual frames would be extracted
        and then these frames will be used for face extraction process.
        """
        # video_path = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\videos\\interview_woman.mp4"
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        video_path, _ = qwt.QFileDialog.getOpenFileName(
            self,
            'Select video file',
            "data/videos",
            "Video files (*.mp4)",
            options=options)

        if video_path:
            logger.info(
                f'{self.data_type.value} video selected from: {video_path}.')

            self.video_player.video_selection.emit(video_path)
            video_name = video_path.split(os.sep)[-1]
            self.preview_label.setText(f'Preview of the: {video_name}')
            self.preview_widget.setCurrentWidget(self.video_player_wgt)
            self.video_path = video_path

        else:
            logger.warning('No directory selected.')

    def select_pictures(self):
        """Select directory with faces which would be used for face
        extraction process.
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self,
        #     "getExistingDirectory",
        #     "./"
        # )
        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\gen_faces"

        if directory:

            logger.info(f'Selected pictures folder: {directory}.')

            self.preview_widget.setCurrentWidget(self.picture_viewer)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                logger.warning(
                    f'No supported pictures were found in: {directory}.'
                )

            else:
                self.picture_viewer.pictures_added_sig.emit(image_paths)

                logger.info(
                    f'Selected {directory} as a ' +
                    f'{self.data_type.value.lower()} data directory.' +
                    f' This directory contains {len(image_paths)} ' +
                    'supported pictures.'
                )

                self.preview_label.setText(
                    f'Preview of pictures in {directory} directory.'
                )

                self.data_directory = directory

                if self.data_type == DATA_TYPE.INPUT:
                    self.signals[SIGNAL_OWNER.INPUT_DATA_DIRECTORY].emit(
                        directory
                    )
                else:
                    self.signals[SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY].emit(
                        directory
                    )

        else:
            logger.warning('No directory selected.')

    def select_frames_directory(self):
        """Selects where extracted frames from video will go.
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self, "getExistingDirectory", "./")
        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\gen_faces"
        if directory:
            msg = Messages.CONSOLE_PRINT(
                CONSOLE_MESSAGE_TYPE.LOG,
                f'Selected {self.data_type.value.lower()} ' +
                f'directory: {directory} for extracted frames.'
            )

            logger.info(
                f'Selected {self.data_type.value.lower()} ' +
                f'directory: {directory} for extracted frames.'
            )

            if self.resize_frames_chk.isChecked():
                if self.biggest_frame_dim_value is not None:
                    self.enable_widget(self.extract_frames_btn, True)
            else:
                self.enable_widget(self.extract_frames_btn, True)

            self.data_directory = directory

            if self.data_type == DATA_TYPE.INPUT:
                self.signals[SIGNAL_OWNER.INPUT_DATA_DIRECTORY].emit(
                    directory
                )
            else:
                self.signals[SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY].emit(
                    directory
                )

        else:
            logger.warning('No directory selected.')

    @qtc.pyqtSlot(str)
    def biggest_frame_dim_input_text_changed(self, text: str):
        """Input for biggest picture dimension if user wants to resize
        frames extracted from video.

        Parameters
        ----------
        text : str
            user input
        """
        try:
            num = int(text)
            self.biggest_frame_dim_value = num
            if self.data_directory is not None:
                self.enable_widget(self.extract_frames_btn, True)
        except ValueError:
            self.biggest_frame_dim_value = None
            self.enable_widget(self.extract_frames_btn, False)

    def extract_frames(self):
        """Sends signal to MakeDeepfakePage to start frames extraction
        process for input or output data.
        """
        body_data = {
            BODY_KEY.RESIZE: False,
            BODY_KEY.DATA_TYPE: self.data_type,
            BODY_KEY.VIDEO_PATH: self.video_path,
        }
        if self.resize_frames_chk.isChecked():
            body_data[BODY_KEY.RESIZE] = True
            body_data[BODY_KEY.NEW_SIZE] = self.biggest_frame_dim_value

        msg = Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.DATA_SELECTOR,
            SIGNAL_OWNER.FRAMES_EXTRACTION,
            Body(
                JOB_TYPE.FRAME_EXTRACTION,
                body_data,
            )
        )

        self.signals[SIGNAL_OWNER.FRAMES_EXTRACTION].emit(msg)

    @qtc.pyqtSlot(int)
    def resize_frames_chk_changed(self, value: int):
        """Enables widget for biggest frame dimension input when is checked.

        Parameters
        ----------
        value : int
            new checkbox value
        """
        if value == 2:  # checkbox checked value
            self.enable_widget(self.biggest_frame_dim_input, True)
            if self.biggest_frame_dim_input.text() == '':
                self.enable_widget(self.extract_frames_btn, False)
        else:
            self.enable_widget(self.biggest_frame_dim_input, False)
            self.enable_widget(self.extract_frames_btn, True)
