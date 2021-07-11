import PyQt5.QtWidgets as qwt

from gui.pages.page import Page, CONSOLE_MESSAGE_TYPE
from gui.widgets.video_player import VideoPlayer
from gui.templates.make_deepfake_page import Ui_make_deepfake_page

from names import MAKE_DEEPFAKE_PAGE_NAME, MAKE_DEEPFAKE_PAGE_TITLE, START_PAGE_NAME


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name=MAKE_DEEPFAKE_PAGE_NAME, *args, **kwargs)

        self.setupUi(self)

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.back_btn.clicked.connect(self.goto_start_page)

        self.select_video_btn.clicked.connect(self.select_video)

        self.tabs.setTabEnabled(1, False)  # disable tab 2
        self.tabs.setTabEnabled(2, False)  # disable tab 3

        self.video_player = VideoPlayer()
        self.tab_1_layout.addWidget(self.video_player)
        self.tab_1_layout.addWidget(qwt.QLabel(text='labels'))
        self.video_player.hide()

    def goto_start_page(self):
        self.show_menubar(False)
        self.show_console(False)
        self.goto(START_PAGE_NAME)

    def select_video(self):
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        fileName, _ = qwt.QFileDialog.getOpenFileName(
            self, 'Select video file', "data/videos", "Video files (*.mp4)", options=options)
        if fileName:
            self.video_player.video_selection.emit(fileName)
            self.video_player.show()
            self.print('Loaded video: ' + fileName, CONSOLE_MESSAGE_TYPE.ERROR)
        else:
            self.print('No video selected', CONSOLE_MESSAGE_TYPE.WARNING)

        self.print('No video selected', CONSOLE_MESSAGE_TYPE.LOG)
        self.print('No video selected', CONSOLE_MESSAGE_TYPE.INFO)
