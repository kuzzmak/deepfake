import logging
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Dict, Tuple, Union

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.app_config import APP_CONFIG
from core.exception import SeleniumNotFoundError
import core.scraper.google_images_scraper as gis
from core.worker import FramesExtractionWorker, Worker
from enums import CONNECTION, JOB_TYPE, LAYOUT, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    Button,
    GroupBox,
    HWidget,
    InfoIconButton,
    PlayIcon,
    StopIcon,
    VWidget,
    VerticalSpacer,
)
from gui.widgets.image_viewer.image_viewer import ImageViewer
from gui.widgets.video_player import VideoPlayer
from utils import parse_number

logger = logging.getLogger(__name__)


class GoogleImagesScraperWorker(qtc.QObject):

    _stop_sig = qtc.pyqtSignal()
    started = qtc.pyqtSignal()
    finished = qtc.pyqtSignal()
    new_image_sig = qtc.pyqtSignal(list)

    def __init__(
        self,
        search_terms: List[str],
        save_directory: Union[str, Path],
    ) -> None:
        super().__init__()

        self._search_terms = search_terms
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        self._save_directory = save_directory

        self._sc = gis.GoogleImagesScraper()
        self._stop_sig.connect(self._sc.stop_sig)

    def stop(self) -> None:
        self._stop_sig.emit()

    def run(self) -> None:
        self.started.emit()
        save_path = self._save_directory / '_'.join(self._search_terms)
        self._sc.run(
            keywords=self._search_terms,
            save_directory=save_path,
            n_suggested=10,
            new_image_sig=self.new_image_sig,
        )
        self._sc.close_driver()
        self.finished.emit()


class DataTab(BaseWidget):

    stop_google_images_scraper_worker_sig = qtc.pyqtSignal()
    stop_frames_extraction_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._video_path = None
        self._frames_directory = None
        self._gi_dir = APP_CONFIG \
            .app \
            .google_images_scraper \
            .default_save_directory
        self._scraping_running = False
        self._frames_extraction_in_progress = False
        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()

        self.init_ui()

    def init_ui(self):
        layout = qwt.QVBoxLayout()
        center_wgt = qwt.QWidget()
        self.stacked_wgt = qwt.QStackedWidget()
        central_layout = qwt.QVBoxLayout()
        center_wgt.setLayout(central_layout)

        lbl_text = 'This tab provides funcionality for splitting ' + \
            'video into frames or scraping Google images ' + \
            'for data based on search term.'
        central_layout.addWidget(qwt.QLabel(text=lbl_text))

        data_gb = qwt.QGroupBox()
        data_gb.setTitle('Data acquisition methods')
        data_gb_layout = qwt.QHBoxLayout(data_gb)

        central_layout.addWidget(data_gb)
        central_layout.addWidget(self.stacked_wgt)

        data_bg = qwt.QButtonGroup(data_gb)
        data_bg.idPressed.connect(self._data_acquisition_method_selected)

        video_rbtn = qwt.QRadioButton(text='Video', parent=data_gb)
        video_rbtn.setChecked(True)
        data_gb_layout.addWidget(video_rbtn)
        data_bg.addButton(video_rbtn)

        google_images_rbtn = qwt.QRadioButton(
            text='Google images',
            parent=data_gb,
        )
        data_gb_layout.addWidget(google_images_rbtn)
        data_bg.addButton(google_images_rbtn)

        # --- video widget ---
        self.video_wgt = HWidget()
        self.stacked_wgt.addWidget(self.video_wgt)
        self.video_wgt.layout().setContentsMargins(0, 0, 0, 0)

        left_part = VWidget()
        self.video_wgt.layout().addWidget(left_part)
        left_part.layout().setContentsMargins(0, 0, 0, 0)
        left_part.setFixedWidth(190)
        right_part = VWidget()
        self.video_wgt.layout().addWidget(right_part)
        right_part.layout().setContentsMargins(0, 0, 0, 0)

        select_video_gb = GroupBox('Video source')
        left_part.layout().addWidget(select_video_gb)
        select_video_btn = Button('select')
        select_video_gb.layout().addWidget(select_video_btn)
        select_video_btn.clicked.connect(self._select_video)

        self.video_player = VideoPlayer()
        right_part.layout().addWidget(self.video_player)

        frames_gb = GroupBox('Directory for extracted frames')
        left_part.layout().addWidget(frames_gb)
        self.select_output_directory_btn = Button('select')
        frames_gb.layout().addWidget(self.select_output_directory_btn)
        self.select_output_directory_btn.clicked.connect(
            self._select_frames_directory
        )

        n_th_frame_gb = GroupBox('Extract every n-th frame')
        left_part.layout().addWidget(n_th_frame_gb)
        self.n_th_frame_edit = qwt.QLineEdit()
        self.n_th_frame_edit.setText('10')
        self.n_th_frame_edit.setFixedWidth(150)
        n_th_frame_gb.layout().addWidget(self.n_th_frame_edit)

        left_part.layout().addItem(VerticalSpacer())

        self.extraction_btn = Button('start extraction')
        left_part.layout().addWidget(
            self.extraction_btn,
            0,
            qtc.Qt.AlignmentFlag.AlignHCenter,
        )
        self.extraction_btn.setIcon(PlayIcon())
        self.extraction_btn.clicked.connect(self._frames_extraction_op)

        # --- google images scraper widget ---
        self.google_images_wgt = HWidget()
        self.stacked_wgt.addWidget(self.google_images_wgt)
        self.google_images_wgt.layout().setContentsMargins(0, 0, 0, 0)

        left_part_gi = VWidget()
        self.google_images_wgt.layout().addWidget(left_part_gi)
        left_part_gi.setMaximumWidth(250)

        gi_dir_gb = GroupBox('Directory where scraped images will be saved')
        left_part_gi.layout().addWidget(gi_dir_gb)
        select_gi_dir_btn = Button(text='select')
        gi_dir_gb.layout().addWidget(select_gi_dir_btn)
        select_gi_dir_btn.clicked.connect(self._select_gi_dir)

        gi_depth_gb = GroupBox(
            'Suggested results depth',
            LAYOUT.HORIZONTAL,
        )
        left_part_gi.layout().addWidget(gi_depth_gb)
        self.suggested_depth_input = qwt.QLineEdit()
        gi_depth_gb.layout().addWidget(self.suggested_depth_input)
        self.suggested_depth_input.setText(
            str(APP_CONFIG.app.google_images_scraper.suggested_search_depth)
        )
        gi_depth_gb.layout().addWidget(InfoIconButton(
            'Number of first images on the google search results for which ' +
            'suggested images will also be downloaded.'
        ))

        gi_recent_images_num = GroupBox(
            'Number of recent images',
            LAYOUT.HORIZONTAL,
        )
        left_part_gi.layout().addWidget(gi_recent_images_num)
        self.recent_images_num_input = qwt.QLineEdit()
        gi_recent_images_num.layout().addWidget(self.recent_images_num_input)
        self.recent_images_num_input.setText(
            str(APP_CONFIG.app.google_images_scraper.default_page_limit)
        )
        gi_recent_images_num.layout().addWidget(InfoIconButton(
            'How many recent images will be shown in ' +
            'the image viewer to the right.'
        ))

        gi_search_term = GroupBox('Search term')
        left_part_gi.layout().addWidget(gi_search_term)
        self.searach_term_input = qwt.QLineEdit()
        gi_search_term.layout().addWidget(self.searach_term_input)
        self.searach_term_input.setText('donald trump')

        self.scraping_btn = Button('start scraping')
        left_part_gi.layout().addWidget(self.scraping_btn)
        self.scraping_btn.setIcon(PlayIcon())
        self.scraping_btn.clicked.connect(self._scraping_op)

        left_part_gi.layout().addItem(VerticalSpacer())

        right_part_gi = VWidget()
        self.google_images_wgt.layout().addWidget(right_part_gi)

        self.gi_image_viewer = ImageViewer(
            icon_size=(128, 128),
            page_limit=APP_CONFIG.app.google_images_scraper.default_page_limit,
            disable_context_menu=True,
        )
        right_part_gi.layout().addWidget(self.gi_image_viewer)

        layout.addWidget(center_wgt)
        self.setLayout(layout)

    @qtc.pyqtSlot(int)
    def _data_acquisition_method_selected(self, val: int) -> None:
        if val == -2:
            self.stacked_wgt.setCurrentWidget(self.video_wgt)
        elif val == -3:
            self.stacked_wgt.setCurrentWidget(self.google_images_wgt)

    @qtc.pyqtSlot()
    def _select_video(self) -> None:
        """Select video from which individual frames would be extracted
        and then these frames will be used for face detection process.
        """
        video_path, _ = qwt.QFileDialog.getOpenFileName(
            self,
            'Select video file',
            "data/videos",
            "Video files (*.mp4)",
        )
        if video_path:
            logger.info(f'Selected video: {video_path}.')
            self.video_player.video_selection.emit(video_path)
            # video_name = video_path.split(os.sep)[-1]
            # self.preview_label.setText(f'Preview of the: {video_name}')
            # self.preview_widget.setCurrentWidget(self.video_player_wgt)
            self._video_path = video_path
        else:
            logger.warning('No directory selected.')

    @qtc.pyqtSlot()
    def _select_gi_dir(self) -> None:
        dir = qwt.QFileDialog.getExistingDirectory(
            self,
            'Select directory for scraped images',
        )
        if not dir:
            logger.warning('No directory selected.')
            return

        self._gi_dir = Path(dir)

    @qtc.pyqtSlot()
    def _scraping_op(self) -> None:
        """pyqtSlot for starting or stopping google images scraper worker.
        """
        # TODO fix:
        # Traceback(most recent call last):
        # File "C:\Users\tonkec\Documents\deepfake\gui\pages\make_deepfake_page\tabs\data_tab.py", line 63, in run
        #     new_image_sig=self.new_image_sig,
        # File "C:\Users\tonkec\Documents\deepfake\core\scraper\google_images_scraper.py", line 248, in run
        #     images = self._find_all_images(skip_main_images)
        # File "C:\Users\tonkec\Documents\deepfake\core\scraper\google_images_scraper.py", line 155, in _find_all_images
        #     if self._exists_more():
        # File "C:\Users\tonkec\Documents\deepfake\core\scraper\google_images_scraper.py", line 111, in _exists_more
        #     'div.YstHxe > input',
        # File "C:\Users\tonkec\Documents\deepfake\env\lib\site-packages\selenium\webdriver\remote\webdriver.py", line 857, in find_element
        #     'value': value})['value']
        # File "C:\Users\tonkec\Documents\deepfake\env\lib\site-packages\selenium\webdriver\remote\webdriver.py", line 428, in execute
        #     self.error_handler.check_response(response)
        # File "C:\Users\tonkec\Documents\deepfake\env\lib\site-packages\selenium\webdriver\remote\errorhandler.py", line 243, in check_response
        #     raise exception_class(message, screen, stacktrace)
        # selenium.common.exceptions.NoSuchElementException: Message: no such
        # element: Unable to locate element: {"method":"css
        # selector","selector":"div.YstHxe > input"}
        if not self._scraping_running:
            text = self.searach_term_input.text()
            text = text.strip()
            terms = text.split()
            if len(terms) == 0:
                logger.warning(
                    'Search term was not provided, scraping will not start.'
                )
                return

            try:
                worker = GoogleImagesScraperWorker(terms, self._gi_dir)
            except SeleniumNotFoundError:
                logger.warning(
                    'Selenium was not found on system but is neccessary'
                    ' to run scraping, would you like to install it?'
                )

                qmb = qwt.QMessageBox
                res = qmb.question(
                    self,
                    'Warning',
                    'Would you like to install neccesary packages to\n'
                    'run Google Images scraping?',
                    buttons=qmb.StandardButton.Yes | qmb.StandardButton.No,
                )
                if res == qmb.StandardButton.Yes:
                    self.enable_widget(self.scraping_btn, False)
                    import subprocess
                    import sys
                    import importlib
                    try:
                        # TODO: this blocks, make an simple wrapper like
                        # Worker for this kind of tasks
                        subprocess.check_call(
                            [
                                sys.executable,
                                '-m',
                                'pip',
                                'install',
                                'selenium',
                                'webdriver-manager',
                            ]
                        )
                        importlib.reload(gis)
                    except CalledProcessError:
                        self.enable_widget(self.scraping_btn, True)
                        logger.error(
                            'Error happened while installing required packages'
                            ' for Google Images scraping. Check pip package'
                            ' manager.'
                        )
                        return
                else:
                    logger.warning(
                        'Not possible to run web scraping without installing'
                        'required packages.'
                    )
                    self.enable_widget(self.scraping_btn, True)
                    return

            self.enable_widget(self.scraping_btn, True)

            # if execution comes to here, everything should be installed and
            # package with google images scraper reimported
            worker = GoogleImagesScraperWorker(terms, self._gi_dir)
            self.stop_google_images_scraper_worker_sig.connect(worker.stop)
            thread = qtc.QThread()
            worker.moveToThread(thread)
            self._threads[JOB_TYPE.IMAGE_SCRAPING] = (thread, worker)
            thread.started.connect(worker.run)
            worker.started.connect(self._google_images_scraper_worker_started)
            worker.finished.connect(
                self._google_images_scraper_worker_finished
            )
            worker.new_image_sig.connect(self.gi_image_viewer.images_added_sig)
            thread.start()
        else:
            self._stop_scraping_worker()

    def _stop_scraping_worker(self) -> None:
        """Sends signal to google images scraper worker to stop.
        """
        self.stop_google_images_scraper_worker_sig.emit()

    @qtc.pyqtSlot()
    def _google_images_scraper_worker_started(self) -> None:
        """Executes when google images scraper worker starts.
        """
        self._scraping_running = True
        self.scraping_btn.setText('stop scraping')
        self.scraping_btn.setIcon(StopIcon())

    @qtc.pyqtSlot()
    def _google_images_scraper_worker_finished(self) -> None:
        """Waits for google images scraper thread to finish and
        exit gracefully.
        """
        val = self._threads.get(JOB_TYPE.IMAGE_SCRAPING, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.IMAGE_SCRAPING, None)
        self._scraping_running = False
        self.scraping_btn.setText('start scraping')
        self.scraping_btn.setIcon(PlayIcon())

    @qtc.pyqtSlot()
    def _select_frames_directory(self) -> None:
        """Selects directory where extracted frames will be saved.
        """
        dir = qwt.QFileDialog.getExistingDirectory(
            self,
            'Select directory for extracted frames',
        )
        if not dir:
            logger.warning('No directory selected.')
            return
        self._frames_directory = dir

    @qtc.pyqtSlot()
    def _frames_extraction_op(self) -> None:
        """Starts or stops frames extraction process.
        """
        if self._frames_extraction_in_progress:
            self._stop_frames_extraction()
            return
        video_path = self._video_path
        if video_path is None:
            logger.error('No video was selected.')
            return
        frames_directory = self._frames_directory
        if frames_directory is None:
            logger.error('No directory for frames was selected.')
            return
        every_nth = parse_number(self.n_th_frame_edit.text())
        if every_nth is None:
            logger.error(
                'Unable to parse number for every nth ' +
                'frame, must be integer.'
            )
            return
        thread = qtc.QThread()
        worker = FramesExtractionWorker(
            video_path,
            frames_directory,
            every_nth,
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
        )
        self.stop_frames_extraction_sig.connect(
            lambda: worker.conn_q.put(CONNECTION.STOP)
        )
        worker.moveToThread(thread)
        self._threads[JOB_TYPE.FRAMES_EXTRACTION] = (thread, worker)
        thread.started.connect(worker.run)
        worker.started.connect(self._on_frames_extraction_worker_started)
        worker.finished.connect(self._on_frames_extraction_worker_finished)
        thread.start()

    def _stop_frames_extraction(self) -> None:
        """Emits signal to the frames extraction worker to stop.
        """
        logger.info('Requested stop of the frames extraction, please wait...')
        self.stop_frames_extraction_sig.emit()

    @qtc.pyqtSlot()
    def _on_frames_extraction_worker_started(self) -> None:
        """Changed text and icon of the button for frames extraction when
        the process starts.
        """
        self._frames_extraction_in_progress = True
        self.extraction_btn.setText('stop extraction')
        self.extraction_btn.setIcon(StopIcon())

    @qtc.pyqtSlot()
    def _on_frames_extraction_worker_finished(self) -> None:
        """Changed text and icon of the button for frames extraction when
        the process ends.
        """
        val = self._threads.get(JOB_TYPE.FRAMES_EXTRACTION, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.FRAMES_EXTRACTION, None)
        self._frames_extraction_in_progress = False
        self.extraction_btn.setText('start extraction')
        self.extraction_btn.setIcon(PlayIcon())
