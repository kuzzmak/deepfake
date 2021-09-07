from datetime import datetime
import logging

from console import Console
from enums import Level


DATE_FORMAt = '%Y-%m-%d %H:%M:%S'


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
