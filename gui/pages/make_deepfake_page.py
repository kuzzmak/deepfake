from PyQt5 import QtCore
from PyQt5.QtGui import QTextCursor
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget

from gui.pages.page import Page
from gui.templates.make_deepfake_page import Ui_make_deepfake_page

from names import MAKE_DEEPFAKE_PAGE_NAME, MAKE_DEEPFAKE_PAGE_TITLE, START_PAGE_NAME


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)
        self.name = MAKE_DEEPFAKE_PAGE_NAME

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.back_btn.clicked.connect(self.goto_start_page)

        self.pushButton.clicked.connect(self.write)

        self.select_video_btn.clicked.connect(self.select_video)

        videoWidget = QVideoWidget()
        videoWidget.setFixedSize(100, 100)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.tab_1_layout.addWidget(videoWidget)

    def goto_start_page(self):
        self.goto(START_PAGE_NAME)

    def write(self):
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText('kurac baho moj' + '\n')
        self.console.moveCursor(QTextCursor.End)

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, 'Select video file', "data/videos", "Video files (*.mp4)", options=options)
        if fileName:
            print(fileName)
            self.mediaPlayer.setMedia(
                QMediaContent(QtCore.QUrl.fromLocalFile(fileName)))

            self.mediaPlayer.play()
