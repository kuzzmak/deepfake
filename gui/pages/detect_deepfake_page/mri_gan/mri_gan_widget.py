import logging
from multiprocessing import Lock
from typing import Dict, Optional, Tuple

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt
from gui.pages.detect_deepfake_page.mri_gan.train_df_detector_widget import \
    TrainDeepfakeDetectorWidget
from gui.pages.detect_deepfake_page.mri_gan.train_mri_gan_widget import \
    TrainMRIGANWidget
from core.worker import (
    CropFacesWorker,
    GenerateMRIDatasetWorker,
    LandmarkExtractionWorker,
)
from core.worker.worker import Worker
from enums import CONNECTION, JOB_TYPE, SIGNAL_OWNER
from gui.pages.detect_deepfake_page.model_widget import ModelWidget
from gui.pages.detect_deepfake_page.mri_gan.common import Step
from gui.widgets.common import (
    HWidget,
    PlayIcon,
    StopIcon,
    VWidget,
    VerticalSpacer,
)
from gui.widgets.configure_data_paths_dialog import ConfigureDataPathsDialog
from utils import parse_number


logger = logging.getLogger(__name__)


class MRIGANWidget(ModelWidget):

    stop_landmark_extraction_sig = qtc.pyqtSignal()
    stop_cropping_faces_sig = qtc.pyqtSignal()
    stop_gen_mri_dataset_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals)

        self._init_ui()

        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()
        self._lmrks_extraction_in_progress = False
        self._cropping_faces_in_progress = False
        self._gen_mri_dataset_in_progress = False
        self._lock = Lock()

    def _init_ui(self) -> None:
        ######
        # DATA
        ######
        central_wgt_data_tab = HWidget()
        self.data_tab.layout().addWidget(central_wgt_data_tab)
        central_wgt_data_tab.layout().setContentsMargins(0, 0, 0, 0)

        left_part_data_tab = VWidget()
        central_wgt_data_tab.layout().addWidget(left_part_data_tab)
        left_part_data_tab.setMaximumWidth(460)

        left_part_data_tab.layout().addWidget(qwt.QLabel(text='mri gan model'))

        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.Shape.VLine)
        line.setFrameShadow(qwt.QFrame.Shadow.Sunken)
        central_wgt_data_tab.layout().addWidget(line)

        right_part_data_tab = VWidget()
        central_wgt_data_tab.layout().addWidget(right_part_data_tab)

        right_part_data_tab.layout().addWidget(qwt.QLabel(
            text='deepkafe detection model'
        ))

        right_part_data_tab.layout().addItem(VerticalSpacer())

        ##########################
        # LANDMARK EXTRACTION STEP
        ##########################
        self.lmrks_extraction_step = Step(
            'Landmark extraction',
            'start extraction',
        )
        left_part_data_tab.layout().addWidget(self.lmrks_extraction_step)
        self.lmrks_extraction_step.start_btn.clicked.connect(
            self._extract_landmarks
        )
        self.lmrks_extraction_step.configure_paths_btn.clicked.connect(
            self._configure_ext_lmrks_paths
        )

        #####################
        # CROPPING FACES STEP
        #####################
        self.crop_faces_step = Step('Crop faces', 'start cropping')
        left_part_data_tab.layout().addWidget(self.crop_faces_step)
        self.crop_faces_step.start_btn.clicked.connect(self._crop_faces)
        self.crop_faces_step.configure_paths_btn.clicked.connect(
            self._configure_crop_faces_paths
        )

        ######################
        # GENERATE MRI DATASET
        ######################
        self.gen_mri_dataset_step = Step(
            'Generate MRI dataset',
            'generate dataset',
        )
        left_part_data_tab.layout().addWidget(self.gen_mri_dataset_step)
        self.gen_mri_dataset_step.start_btn.clicked.connect(
            self._gen_mri_dataset
        )
        self.gen_mri_dataset_step.configure_paths_btn.clicked.connect(
            self._configure_generate_mri_dataset_paths
        )

        left_part_data_tab.layout().addItem(VerticalSpacer())

        ##########
        # TRAINING
        ##########
        central_wgt_training_tab = HWidget()
        self.training_tab.layout().addWidget(central_wgt_training_tab)
        central_wgt_training_tab.layout().setContentsMargins(0, 0, 0, 0)

        left_part_training_tab = VWidget()
        central_wgt_training_tab.layout().addWidget(left_part_training_tab)
        left_part_training_tab.setMaximumWidth(460)

        left_part_training_tab.layout().addWidget(qwt.QLabel(
            text='mri gan model'
        ))

        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.Shape.VLine)
        line.setFrameShadow(qwt.QFrame.Shadow.Sunken)
        central_wgt_training_tab.layout().addWidget(line)

        right_part_training_tab = VWidget()
        central_wgt_training_tab.layout().addWidget(right_part_training_tab)

        right_part_training_tab.layout().addWidget(qwt.QLabel(
            text='deepkafe detection model'
        ))

        right_part_training_tab.layout().addItem(VerticalSpacer())

        ###############
        # TRAIN MRI GAN
        ###############
        signals = {
            SIGNAL_OWNER.MESSAGE_WORKER:
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
        }
        train_mri_wgt = TrainMRIGANWidget(signals)
        left_part_training_tab.layout().addWidget(train_mri_wgt)

        #########################
        # TRAIN DEEPFAKE DETECTOR
        #########################
        signals = {
            SIGNAL_OWNER.MESSAGE_WORKER:
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
        }
        train_df_detector = TrainDeepfakeDetectorWidget(signals)
        right_part_training_tab.layout().addWidget(train_df_detector)

    @qtc.pyqtSlot()
    def _extract_landmarks(self) -> None:
        """Initiates landmark extraction process or stops it.
        """
        if self._lmrks_extraction_in_progress:
            self._stop_landmark_extraction()
        else:
            num_proc = parse_number(
                self.lmrks_extraction_step.num_of_instances
            )
            if num_proc is None:
                logger.error(
                    'Unable to parse your input for number of ' +
                    'processes, you should put integer number.'
                )
                return

            thread = qtc.QThread()
            worker = LandmarkExtractionWorker(
                self.lmrks_extraction_step.selected_data_type,
                num_proc,
                self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
            )
            self.stop_landmark_extraction_sig.connect(
                lambda: worker.conn_q.put(CONNECTION.STOP)
            )
            worker.moveToThread(thread)
            self._threads[JOB_TYPE.LANDMARK_EXTRACTION] = (thread, worker)
            thread.started.connect(worker.run)
            worker.started.connect(
                self._on_landmark_extraction_worker_started
            )
            worker.running.connect(
                self._on_landmark_extraction_worker_running
            )
            worker.finished.connect(
                self._on_landmark_extraction_worker_finished
            )
            thread.start()

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_finished(self) -> None:
        """Waits for landmark extraction thread to finish and exit gracefully.
        """
        self._lock.acquire()
        try:
            val = self._threads.get(JOB_TYPE.LANDMARK_EXTRACTION, None)
            if val is not None:
                thread, _ = val
                thread.quit()
                thread.wait()
                self._threads.pop(JOB_TYPE.LANDMARK_EXTRACTION, None)
        finally:
            self._lock.release()
        self.enable_widget(self.lmrks_extraction_step.start_btn, True)
        self.lmrks_extraction_step.start_btn.setIcon(PlayIcon())
        self.lmrks_extraction_step.start_btn.setText('start extraction')
        self._lmrks_extraction_in_progress = False

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_started(self) -> None:
        """Slot which disables button for starting or stopping landmark
        extraction until worker is set up.
        """
        self.enable_widget(self.lmrks_extraction_step.start_btn, False)
        self._lmrks_extraction_in_progress = True

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_running(self) -> None:
        """Slot which enables button for starting or stopping landmark
        extraction and changes icon of the button to the stop icon.
        """
        self.enable_widget(self.lmrks_extraction_step.start_btn, True)
        self.lmrks_extraction_step.start_btn.setIcon(StopIcon())
        self.lmrks_extraction_step.start_btn.setText('stop extraction')

    def _stop_landmark_extraction(self) -> None:
        """Sends signal to stop landmark extraction.
        """
        logger.info('Requested stop of landmark extraction, please wait...')
        self.enable_widget(self.lmrks_extraction_step.start_btn, False)
        self.stop_landmark_extraction_sig.emit()

    @qtc.pyqtSlot()
    def _crop_faces(self) -> None:
        """Initiates or stops cropping faces process.
        """
        if self._cropping_faces_in_progress:
            self._stop_cropping_faces()
        else:
            num_proc = parse_number(
                self.crop_faces_step.num_of_instances
            )
            if num_proc is None:
                logger.error(
                    'Unable to parse your input for number of ' +
                    'processes, you should put integer number.'
                )
                return

            thread = qtc.QThread()
            worker = CropFacesWorker(
                self.crop_faces_step.selected_data_type,
                num_proc,
                self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            )
            self.stop_cropping_faces_sig.connect(
                lambda: worker.conn_q.put(CONNECTION.STOP)
            )
            worker.moveToThread(thread)
            self._threads[JOB_TYPE.CROPPING_FACES] = (thread, worker)
            thread.started.connect(worker.run)
            worker.started.connect(self._on_crop_faces_worker_started)
            worker.running.connect(self._on_crop_faces_worker_running)
            worker.finished.connect(self._on_crop_faces_worker_finished)
            thread.start()

    def _stop_cropping_faces(self) -> None:
        """Sends signal to stop cropping faces.
        """
        logger.info('Requested stop of cropping faces, please wait...')
        self.enable_widget(self.crop_faces_step.start_btn, False)
        self.stop_cropping_faces_sig.emit()

    @qtc.pyqtSlot()
    def _on_crop_faces_worker_started(self) -> None:
        """Disables button for stopping the cropping face process until
        everything is set up and runs correctly.
        """
        self.enable_widget(self.crop_faces_step.start_btn, False)
        self._cropping_faces_in_progress = True

    @qtc.pyqtSlot()
    def _on_crop_faces_worker_running(self) -> None:
        """Updates icon and text of the button which starts of stops process
        of cropping faces.
        """
        self.enable_widget(self.crop_faces_step.start_btn, True)
        self.crop_faces_step.start_btn.setIcon(StopIcon())
        self.crop_faces_step.start_btn.setText('stop cropping faces')

    @qtc.pyqtSlot()
    def _on_crop_faces_worker_finished(self) -> None:
        """Waits for cropping thread to quit, handles button update for text
        and icon.
        """
        self._lock.acquire()
        try:
            val = self._threads.get(JOB_TYPE.CROPPING_FACES, None)
            if val is not None:
                thread, _ = val
                thread.quit()
                thread.wait()
                self._threads.pop(JOB_TYPE.CROPPING_FACES, None)
        finally:
            self._lock.release()
        self.enable_widget(self.crop_faces_step.start_btn, True)
        self.crop_faces_step.start_btn.setIcon(PlayIcon())
        self.crop_faces_step.start_btn.setText('crop faces')
        self._cropping_faces_in_progress = False

    @qtc.pyqtSlot()
    def _gen_mri_dataset(self) -> None:
        """Initiates or stops generation of the mri dataset.
        """
        if self._gen_mri_dataset_in_progress:
            self._stop_gen_mri_dataset()
        else:
            num_proc = parse_number(
                self.gen_mri_dataset_step.num_of_instances
            )
            if num_proc is None:
                logger.error(
                    'Unable to parse your input for number of ' +
                    'processes, you should put integer number.'
                )
                return

            thread = qtc.QThread()
            worker = GenerateMRIDatasetWorker(
                self.gen_mri_dataset_step.selected_data_type,
                num_proc,
                self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            )
            self.stop_gen_mri_dataset_sig.connect(
                lambda: worker.conn_q.put(CONNECTION.STOP)
            )
            worker.moveToThread(thread)
            self._threads[JOB_TYPE.GENERATE_MRI_DATASET] = (thread, worker)
            thread.started.connect(worker.run)
            worker.started.connect(self._on_gen_mri_dataset_worker_started)
            worker.running.connect(self._on_gen_mri_dataset_worker_running)
            worker.finished.connect(self._on_gen_mri_dataset_worker_finished)
            thread.start()

    def _stop_gen_mri_dataset(self) -> None:
        """Sends signal to stop generation of the mri dataset.
        """
        logger.info(
            'Requested stop of the generating mri ' +
            'dataset, please wait...'
        )
        self.enable_widget(self.gen_mri_dataset_step.start_btn, False)
        self.stop_gen_mri_dataset_sig.emit()

    @qtc.pyqtSlot()
    def _on_gen_mri_dataset_worker_started(self) -> None:
        """Disables button for stopping worker until setup is done.
        """
        self.enable_widget(self.gen_mri_dataset_step.start_btn, False)
        self._gen_mri_dataset_in_progress = True

    @qtc.pyqtSlot()
    def _on_gen_mri_dataset_worker_running(self) -> None:
        """Changes icon and text of the button which is used to start or stop
        generation of the mri dataset.
        """
        self.enable_widget(self.gen_mri_dataset_step.start_btn, True)
        self.gen_mri_dataset_step.start_btn.setIcon(StopIcon())
        self.gen_mri_dataset_step.start_btn.setText('stop generating')

    @qtc.pyqtSlot()
    def _on_gen_mri_dataset_worker_finished(self) -> None:
        """Waits for gen mri worker to quit in order to shutdown threads or
        processes gracefully.
        """
        self._lock.acquire()
        try:
            val = self._threads.get(JOB_TYPE.GENERATE_MRI_DATASET, None)
            if val is not None:
                thread, _ = val
                thread.quit()
                thread.wait()
                self._threads.pop(JOB_TYPE.GENERATE_MRI_DATASET, None)
        finally:
            self._lock.release()
        self.enable_widget(self.gen_mri_dataset_step.start_btn, True)
        self.gen_mri_dataset_step.start_btn.setIcon(PlayIcon())
        self.gen_mri_dataset_step.start_btn.setText('generate dataset')
        self._gen_mri_dataset_in_progress = False

    @qtc.pyqtSlot()
    def _configure_ext_lmrks_paths(self) -> None:
        """Shows dialog for configuring paths for landmark extraction process.
        """
        keys = [
            *ConfigureDataPathsDialog.dfdc_data_path_all,
            *ConfigureDataPathsDialog.dfdc_landmarks_path_all,
        ]
        labels = [
            'DFDC train dataset path',
            'DFDC valid dataset path',
            'DFDC test dataset path',
            'DFDC train landmarks path',
            'DFDC valid landmarks path',
            'DFDC test landmarks path',
        ]
        dialog = ConfigureDataPathsDialog(keys, labels)
        dialog.exec()

    @qtc.pyqtSlot()
    def _configure_crop_faces_paths(self) -> None:
        """Shows dialog for configuring paths for face cropping process.
        """
        keys = [
            *ConfigureDataPathsDialog.dfdc_data_path_all,
            *ConfigureDataPathsDialog.dfdc_landmarks_path_all,
            *ConfigureDataPathsDialog.dfdc_crop_faces_path_all,
        ]
        labels = [
            'DFDC train dataset path',
            'DFDC valid dataset path',
            'DFDC test dataset path',
            'DFDC train landmarks path',
            'DFDC valid landmarks path',
            'DFDC test landmarks path',
            'DFDC cropped faces train path',
            'DFDC cropped faces valid path',
            'DFDC cropped faces test path',
        ]
        dialog = ConfigureDataPathsDialog(keys, labels)
        dialog.exec()

    @qtc.pyqtSlot()
    def _configure_generate_mri_dataset_paths(self) -> None:
        """Show dialog for configuring paths for generating MRI dataset.
        """
        keys = [
            ConfigureDataPathsDialog.mri_metadata_csv_path,
            *ConfigureDataPathsDialog.dfdc_data_path_all,
            ConfigureDataPathsDialog.dfdc_mri_path,
            *ConfigureDataPathsDialog.dfdc_crop_faces_path_all,
        ]
        labels = [
            'DFDC mri metadata csv path',
            'DFDC train dataset path',
            'DFDC valid dataset path',
            'DFDC test dataset path',
            'DFDF mri path',
            'DFDC cropped faces train path',
            'DFDC cropped faces valid path',
            'DFDC cropped faces test path',
        ]
        dialog = ConfigureDataPathsDialog(keys, labels)
        dialog.exec()
