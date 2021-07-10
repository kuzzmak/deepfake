from gui.pages.page import Page
from gui.templates.start_page import Ui_start_page

from names import START_PAGE_NAME, MAKE_DEEPFAKE_PAGE_NAME, START_PAGE_TITLE


class StartPage(Page, Ui_start_page):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)
        self.page_name = START_PAGE_NAME

        self.setWindowTitle(START_PAGE_TITLE)

        self.make_deepfake_btn.clicked.connect(self.goto_make_deepfake)

    def goto_make_deepfake(self):
        self.goto(MAKE_DEEPFAKE_PAGE_NAME)
