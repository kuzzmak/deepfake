from gui.widgets.picture_viewer import PictureViewer
from names import MAKE_DEEPFAKE_PAGE_NAME, MAKE_DEEPFAKE_PAGE_TITLE
from gui.pages.page import Page
from gui.templates.make_deepfake_page_2 import Ui_make_deepfake_page
import PyQt5.QtWidgets as qwt


class MakeDeepfakePage2(Page, Ui_make_deepfake_page):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name='make_deepfake_page_2', *args, **kwargs)

        self.setupUi(self)

        self.pv = PictureViewer()
        self.preview_widget.addWidget(self.pv)

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.select_pictures_btn.clicked.connect(self.select_pictures)

    def select_pictures(self):
        directory = qwt.QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./")
        if directory:
            self.preview_widget.setCurrentWidget(self.pv)
            # self.preview_widget = PictureViewer()
            # self.preview_widget.repaint()
            # self.add_picture_viewer(directory)

    def add_picture_viewer(self, folder_path: str):
        self.preview_widget = PictureViewer()
        
        
