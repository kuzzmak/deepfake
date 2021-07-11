from gui.widgets.picture_viewer import PictureViewer
from names import MAKE_DEEPFAKE_PAGE_TITLE
from gui.pages.page import CONSOLE_MESSAGE_TYPE, Page
from gui.templates.make_deepfake_page_2 import Ui_make_deepfake_page
import PyQt5.QtWidgets as qwt
import os


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
            images = os.listdir(directory)
            curr_dir = os.path.abspath(directory)
            image_paths = [os.path.join(curr_dir, x) for x in images]
            for img_path in image_paths:
                self.pv.picture_added_sig.emit(img_path)
            # self.preview_widget = PictureViewer()
            # self.preview_widget.repaint()
            # self.add_picture_viewer(directory)
            message = 'Loaded: {} images from: {}'.format(len(images), curr_dir)
            self.print(message, CONSOLE_MESSAGE_TYPE.INFO)

    def add_picture_viewer(self, folder_path: str):
        self.preview_widget = PictureViewer()
        
        
