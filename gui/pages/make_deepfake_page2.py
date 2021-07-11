from gui.widgets.picture_viewer import PictureViewer
from names import MAKE_DEEPFAKE_PAGE_TITLE
from gui.pages.page import CONSOLE_MESSAGE_TYPE, Page
from gui.templates.make_deepfake_page_2 import Ui_make_deepfake_page
import PyQt5.QtWidgets as qwt
import os
from utils import get_file_paths_from_dir


class MakeDeepfakePage2(Page, Ui_make_deepfake_page):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name='make_deepfake_page_2', *args, **kwargs)

        self.setupUi(self)

        self.pv = PictureViewer()
        self.preview_widget.addWidget(self.pv)

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.select_pictures_btn.clicked.connect(self.select_pictures)

    def select_pictures(self):
        directory = qwt.QFileDialog.getExistingDirectory(
            self, "getExistingDirectory", "./")
        if directory:
            self.preview_widget.setCurrentWidget(self.pv)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                self.print(f'No images were found in: {directory}.', CONSOLE_MESSAGE_TYPE.WARNING)
            else:
                for img_path in image_paths:
                    self.pv.picture_added_sig.emit(img_path)

                message = 'Loaded: {} images from: {}.'.format(
                    len(image_paths), directory)
                self.print(message, CONSOLE_MESSAGE_TYPE.LOG)
        else:
            self.print('No folder was selected.', CONSOLE_MESSAGE_TYPE.WARNING)
