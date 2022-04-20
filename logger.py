from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import sys
from typing import TextIO

from console import Console
from enums import LEVEL
from variables import LONG_DATE_FORMAT


_root = Path(__name__).parent
_core_worker = Path('core') / 'worker'
_p = _root / _core_worker
_workers = list(_p.glob('*.py'))
_restricted_workers = ['worker', 'worker_with_pool', 'mri_gan_worker']
_workers = list(filter(lambda w: w.stem not in _restricted_workers, _workers))
WORKER_LOGGERS = ['.'.join([*_core_worker.parts, w.stem]) for w in _workers]


class GuiHandler(logging.Handler):
    """Custom logging handler which should receive records from different
    loggers, format the message and send it to the gui console.
    """

    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord):
        date = datetime.fromtimestamp(record.created).strftime(
            LONG_DATE_FORMAT
        )
        level = LEVEL[record.levelname]
        name = record.name
        msg = record.message
        source_type = 'worker' if record.name in WORKER_LOGGERS \
            else 'widget'
        Console.print(date, name, source_type, level, msg)


class ConsoleHandler(logging.StreamHandler):

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        super().__init__(stream)

    def emit(self, record):
        try:
            record.source_type = 'worker' if record.name in WORKER_LOGGERS \
                else 'widget'
            msg = self.format(record)
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class FileHandler(TimedRotatingFileHandler):

    def __init__(
        self,
        filename,
        when='h',
        interval=1,
        backupCount=0,
        encoding=None,
        delay=False,
        utc=False,
        atTime=None,
    ) -> None:
        super().__init__(
            filename,
            when,
            interval,
            backupCount,
            encoding,
            delay,
            utc,
            atTime,
        )

    def emit(self, record):
        try:
            if self.shouldRollover(record):
                self.doRollover()
            record.source_type = 'worker' if record.name in WORKER_LOGGERS \
                else 'widget'
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)
