import logging
import logging.config
from pathlib import Path
import yaml

from app import App
import config

if __name__ == '__main__':
    conf_path = Path('').absolute() / 'logging_config.yaml'
    with open(conf_path, 'r') as f:
        _conf = yaml.safe_load(f.read())
    logging.config.dictConfig(_conf)
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
    config.refresh_config()

    app = App()
    app.gui()
