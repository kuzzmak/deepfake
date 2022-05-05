import logging
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import PyQt6.QtCore as qtc

from core.worker import Worker
from enums import JOB_NAME, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages

logger = logging.getLogger(__name__)


class FramesExtractionWorker(Worker):
    """Worker used to extract single frames from the video.

    Parameters
    ----------
    video_path : Union[str, Path]
        path to the video file
    frames_directory : Union[str, Path]
        directory where extracted frames will be saved
    every_nth : int, optional
        extract every n-th frame, by default 10
    message_worker_sig : Optional[qtc.pyqtSignal], optional
        signal to the message worker, by default None
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        frames_directory: Union[str, Path],
        every_nth: int = 10,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        if isinstance(video_path, str):
            self._video_path = Path(video_path)
        else:
            self._video_path = video_path
        self._every_nth = every_nth
        if isinstance(frames_directory, str):
            self._frames_directory = Path(frames_directory)
        else:
            self._frames_directory = frames_directory

    def run_job(self) -> None:
        logger.info(
            f'Frames extraction started on video {str(self._video_path)}.'
        )
        vidcap = cv.VideoCapture(str(self._video_path))
        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT)) // \
            self._every_nth
        logger.info(f'There will be total of {total_frames} extracted.')

        msg = Messages.CONFIGURE_WIDGET(
            SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
            WIDGET.JOB_PROGRESS,
            'setMaximum',
            [total_frames],
            JOB_NAME.FRAMES_EXTRACTION,
        )
        self.send_message(msg)

        success, image = vidcap.read()
        count = 0
        while success:
            if self.should_exit():
                logger.info('Frames extraction worker received stop signal.')
                break

            path = self._frames_directory / f'frame_{count}.png'
            if not path.exists():
                cv.imwrite(str(path), image)

            self.report_progress(
                SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
                JOB_TYPE.FRAMES_EXTRACTION,
                count,
                total_frames
            )
            count += 1

            for _ in range(self._every_nth - 1):
                success, image = vidcap.read()
            success, image = vidcap.read()

        logger.info('Frames extraction finished.')
