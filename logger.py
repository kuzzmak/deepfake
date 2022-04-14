from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from typing import TextIO

from console import Console
from enums import Level


DATE_FORMAt = '%Y-%m-%d %H:%M:%S'

WORKERS = ['LandmarkExtraction']


class GuiHandler(logging.Handler):
    """Custom logging handler which should receive records from different
    loggers, format the message and send it to the gui console.
    """

    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord):
        date = datetime.fromtimestamp(record.created).strftime(DATE_FORMAt)
        level = Level[record.levelname]
        name = record.name
        msg = record.message
        source_type = 'worker' if record.name in WORKERS \
            else 'widget'
        Console.print(date, name, source_type, level, msg)


class ConsoleHandler(logging.StreamHandler):

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        super().__init__(stream)

    def emit(self, record):
        try:
            record.source_type = 'worker' if record.name in WORKERS \
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
            record.source_type = 'worker' if record.name in WORKERS \
                else 'widget'
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)
