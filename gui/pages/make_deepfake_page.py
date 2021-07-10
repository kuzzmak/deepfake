import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.widgets.video_player import VideoPlayer
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

        # videoWidget = QVideoWidget()
        # videoWidget.setFixedSize(100, 100)
        # self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.mediaPlayer.setVideoOutput(videoWidget)

        self.video_player = VideoPlayer()
        self.tab_1_layout.addWidget(self.video_player)
        self.tab_1_layout.addWidget(qwt.QLabel(text='labels'))

    def goto_start_page(self):
        self.goto(START_PAGE_NAME)

    def write(self):
        self.console.moveCursor(qtg.QTextCursor.End)
        self.console.insertPlainText('kurac baho moj' + '\n')
        self.console.moveCursor(qtg.QTextCursor.End)

    def select_video(self):
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        fileName, _ = qwt.QFileDialog.getOpenFileName(
            self, 'Select video file', "data/videos", "Video files (*.mp4)", options=options)
        if fileName:
            # print(fileName)
            # self.mediaPlayer.setMedia(
            #     QMediaContent(QtCore.QUrl.fromLocalFile(fileName)))

            # self.mediaPlayer.play()
            self.video_player.video_selection.emit(fileName)
