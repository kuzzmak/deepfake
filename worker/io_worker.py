import logging
from typing import List

import PyQt5.QtCore as qtc

from core.face import Face
from serializer.face_serializer import FaceSerializer

logger = logging.getLogger(__name__)


class Worker(qtc.QObject):

    finished = qtc.pyqtSignal()

    def __init__(self, faces: List[Face], metadata_path: str) -> None:
        super().__init__()
        self.faces = faces
        self.metadata_path = metadata_path

    @qtc.pyqtSlot()
    def run(self):
        logger.info(
            f'Saving {len(self.faces)} faces metadata objects. ' +
            'Please wait.'
        )
        for face in self.faces:
            FaceSerializer.save(face, self.metadata_path)
        logger.info('Finished saving.')
        self.finished.emit()
