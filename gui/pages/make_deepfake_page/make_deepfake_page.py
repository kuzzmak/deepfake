from gui.pages.make_deepfake_page.data_tab import DataTab
from gui.pages.make_deepfake_page.detection_algorithm_tab import DetectionAlgorithmTab
import os

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.templates.make_deepfake_page import Ui_make_deepfake_page
from gui.widgets.video_player import VideoPlayer
from gui.widgets.picture_viewer import PictureViewer

from message.message import (
    ConsolePrintMessageBody,
    FaceDetectionMessageBody,
    FrameExtractionMessageBody,
    Message,
)

from enums import (
    CONSOLE_MESSAGE_TYPE,
    FACE_DETECTION_ALGORITHM,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)

from names import (
    MAKE_DEEPFAKE_PAGE_NAME,
    MAKE_DEEPFAKE_PAGE_TITLE,
)

from utils import get_file_paths_from_dir

from resources.icons import icons

no_foler_selected_msg = Message(
    MESSAGE_TYPE.REQUEST,
    ConsolePrintMessageBody(
        CONSOLE_MESSAGE_TYPE.WARNING,
        'No folder was selected.'
    )
)


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name=MAKE_DEEPFAKE_PAGE_NAME, *args, **kwargs)

        self.data_directory = ''
        self.faces_directory = ''
        self.biggest_frame_dim_value = None

        self.setupUi(self)
        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.picture_viewer_tab_1 = PictureViewer()
        self.preview_widget.addWidget(self.picture_viewer_tab_1)

        # --- video widget with faces directory selection ---
        self.video_player = VideoPlayer()

        self.central_widget = qwt.QWidget()
        central_layout = qwt.QHBoxLayout()
        self.central_widget.setLayout(central_layout)
        central_layout.addWidget(self.video_player)

        self.frame_extraction_gb = qwt.QGroupBox()
        self.frame_extraction_gb.setTitle('Select destination directory ' +
                                          'for extracted frames from video')
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Maximum, qwt.QSizePolicy.Minimum)
        self.frame_extraction_gb.setSizePolicy(size_policy)
        # self.frame_extraction_gb.setMaximumWidth(200)

        frame_extraction_part_layout = qwt.QVBoxLayout(
            self.frame_extraction_gb)

        select_frames_directory = qwt.QPushButton(text='Select')
        select_frames_directory.clicked.connect(self.select_frames_directory)
        frame_extraction_part_layout.addWidget(select_frames_directory)

        # --- resize frames part of the window ---
        resize_frames_chk = qwt.QCheckBox(text='Resize frames')
        resize_frames_chk.stateChanged.connect(self.resize_frames_chk_changed)
        frame_extraction_part_layout.addWidget(resize_frames_chk)
        self.dim_wgt = qwt.QWidget()
        self.enable_widget(self.dim_wgt, False)
        dim_wgt_layout = qwt.QHBoxLayout()
        self.dim_wgt.setLayout(dim_wgt_layout)
        dim_wgt_layout.addWidget(qwt.QLabel(text='Biggest frame dimension: '))
        self.biggest_frame_dim_input = qwt.QLineEdit()
        self.biggest_frame_dim_input.textChanged.connect(
            self.biggest_frame_dim_input_text_changed)
        dim_wgt_layout.addWidget(self.biggest_frame_dim_input)
        frame_extraction_part_layout.addWidget(self.dim_wgt)

        self.extract_frames_btn = qwt.QPushButton(text='Extract frames')
        self.extract_frames_btn.clicked.connect(self.extract_frames)
        self.enable_widget(self.extract_frames_btn, False)

        frame_extraction_part_layout.addWidget(self.extract_frames_btn)

        spacer = qwt.QSpacerItem(
            40,
            20,
            qwt.QSizePolicy.Preferred,
            qwt.QSizePolicy.MinimumExpanding)
        frame_extraction_part_layout.addItem(spacer)

        central_layout.addWidget(self.frame_extraction_gb)
        self.preview_widget.addWidget(self.central_widget)

        # until pictures or video is selected, page with face detection
        # is disabled
        self.enable_detection_algorithm_tab(False)

        self.select_pictures_btn.clicked.connect(self.select_pictures)
        self.select_video_btn.clicked.connect(self.select_video)
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.start_detection_btn.setIcon(qtg.QIcon(qtg.QPixmap(':/play.svg')))
        self.enable_widget(self.start_detection_btn, False)
        self.select_faces_directory_btn.clicked.connect(
            self.select_faces_directory
        )

        self.picture_viewer_tab_2 = PictureViewer()
        self.image_viewer_layout.addWidget(self.picture_viewer_tab_2)

        data_tab = DataTab()
        data_tab.add_signal(
            self.signals[SIGNAL_OWNER.CONOSLE], SIGNAL_OWNER.CONOSLE)
        self.tab_widget.addTab(data_tab, 'Data')
        self.tab_widget.addTab(DetectionAlgorithmTab(), 'Detection algorithm')

    @qtc.pyqtSlot(int)
    def resize_frames_chk_changed(self, value: int):
        """Enables widget for biggest frame dimension input when is checked.

        Parameters
        ----------
        value : int
            new checkbox value
        """
        if value == 2:  # checkbox checked value
            # enable widget with dimension input
            self.enable_widget(self.dim_wgt, True)
            # in order to be able to start frames extraction, directory
            # must be selected and some valid value for resized frame
            # dimension must be inputed
            if self.data_directory != '' and \
                    self.biggest_frame_dim_value is not None:
                self.enable_widget(self.extract_frames_btn, True)
            else:
                self.enable_widget(self.extract_frames_btn, False)
        else:
            self.enable_widget(self.dim_wgt, False)
            self.enable_widget(self.extract_frames_btn, True)

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
            self.enable_widget(self.extract_frames_btn, True)
        except ValueError:
            self.biggest_frame_dim_value = None
            self.enable_widget(self.extract_frames_btn, False)

    def progress_value_changed(self, value: int):
        if value == 100:
            msg = qwt.QMessageBox(self)
            msg.setIcon(qwt.QMessageBox.Information)
            msg.setText("Face extraction successful.")
            msg.setInformativeText("Extracted faces are shown below.")
            msg.setWindowTitle("Face extraction information")
            msg.setStandardButtons(qwt.QMessageBox.Ok)
            msg.exec_()

    def set_preview_label_text(self, text: str):
        self.preview_label.setText(text)

    def enable_detection_algorithm_tab(self, enable: bool):
        self.tab_widget.setTabEnabled(1, enable)

    def start_detection(self):
        """Initiates process of face detection and extraction from
        selected directory.
        """
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            FaceDetectionMessageBody(
                self.faces_directory,
                'C:\\Users\\tonkec\\Documents\\deepfake\\data\\weights\\s3fd\\s3fd.pth',
                FACE_DETECTION_ALGORITHM.S3FD
            )
        )
        self.send_message(msg)

    def extract_frames(self):
        """Starts process of extracting frames from video.
        """
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            ConsolePrintMessageBody(
                CONSOLE_MESSAGE_TYPE.LOG,
                'Started frame extraction.'
            )
        )
        self.send_message(msg)

        msg = Message(
            MESSAGE_TYPE.REQUEST,
            FrameExtractionMessageBody(
                self.video_path,
                self.data_directory,
                True,
                self.biggest_frame_dim_value,
                'jpg'
            )
        )
        self.send_message(msg)

    def select_faces_directory(self):
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self, "getExistingDirectory", "./")
        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\gen_faces"
        if directory:
            self.selected_faces_directory_label.setText(directory)
            self.enable_widget(self.start_detection_btn, True)

            msg = Message(
                MESSAGE_TYPE.REQUEST,
                ConsolePrintMessageBody(
                    CONSOLE_MESSAGE_TYPE.INFO,
                    f'Selected directory for extracted faces: {directory}.'
                )
            )

            self.faces_directory = directory

        else:

            msg = no_foler_selected_msg

        self.send_message(msg)

    def select_frames_directory(self):
        """Select directory in which extracted frames from video will go.
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self,
        #     "getExistingDirectory",
        #     "./")

        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\gen_faces"

        if directory:
            self.data_directory = directory
            msg = Message(
                MESSAGE_TYPE.REQUEST,
                ConsolePrintMessageBody(
                    CONSOLE_MESSAGE_TYPE.INFO,
                    'Directory in which extracted frames will go ' +
                    f'selected: {directory}.'
                )
            )

            self.enable_widget(self.extract_frames_btn, True)

        else:
            msg = no_foler_selected_msg

        self.send_message(msg)

    def select_video(self):
        """Select video from which individual frames would be extracted
        and then these frames are used for face extraction process.
        """
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        # video_path, _ = qwt.QFileDialog.getOpenFileName(
        #     self,
        #     'Select video file',
        #     "data/videos",
        #     "Video files (*.mp4)",
        #     options=options)

        video_path = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\videos\\interview_woman.mp4"

        if video_path:
            self.video_path = video_path

            self.video_player.video_selection.emit(video_path)
            self.preview_widget.setCurrentWidget(self.central_widget)

            video_name = os.path.splitext(os.path.basename(video_path))[0]

            self.set_preview_label_text(
                'Preview of: ' + video_name + ' video.')

            msg = Message(
                MESSAGE_TYPE.REQUEST,
                ConsolePrintMessageBody(
                    CONSOLE_MESSAGE_TYPE.LOG,
                    f'Loaded video from: {video_path}'))

            self.enable_detection_algorithm_tab(True)

        else:
            msg = no_foler_selected_msg

        self.send_message(msg)

    def select_pictures(self):
        """Selecting directory with faces which would be used for face
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
            self.preview_widget.setCurrentWidget(self.picture_viewer_tab_1)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                msg = Message(
                    MESSAGE_TYPE.REQUEST,
                    ConsolePrintMessageBody(
                        CONSOLE_MESSAGE_TYPE.WARNING,
                        f'No images were found in: {directory}.'
                    )
                )

                self.enable_detection_algorithm_tab(False)

            else:
                self.picture_viewer_tab_1.pictures_added_sig.emit(image_paths)

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

                self.set_preview_label_text(
                    'Preview of pictures in: ' + directory + ' directory.')

                self.enable_detection_algorithm_tab(True)

        else:

            msg = no_foler_selected_msg

        self.send_message(msg)