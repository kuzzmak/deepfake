import logging
import logging.config
import yaml

from app import App
from configs.app_config import refresh_config
from variables import LOGGING_CONFIG_PATH

if __name__ == '__main__':
    conf_path = LOGGING_CONFIG_PATH
    with open(conf_path, 'r') as f:
        _conf = yaml.safe_load(f.read())
    logging.config.dictConfig(_conf)
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
    refresh_config()

    app = App()
    app.gui()
