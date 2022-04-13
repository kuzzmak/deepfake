from datetime import datetime
import logging
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
        Console.print(date, name, level, msg)


class CustomConsoleHandler(logging.StreamHandler):

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
