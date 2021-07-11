from gui.pages.page import Page
from gui.templates.start_page import Ui_start_page

from names import START_PAGE_NAME, MAKE_DEEPFAKE_PAGE_NAME, START_PAGE_TITLE


class StartPage(Page, Ui_start_page):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, page_name=START_PAGE_NAME, *args, **kwargs)

        self.setupUi(self)

        self.setWindowTitle(START_PAGE_TITLE)

        self.make_deepfake_btn.clicked.connect(self.goto_make_deepfake)

    def goto_make_deepfake(self):
        self.show_toolbars_and_console(True)
        # self.goto(MAKE_DEEPFAKE_PAGE_NAME)
        self.goto('make_deepfake_page_2')
