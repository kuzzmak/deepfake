import logging
import logging.config
import yaml

from app import App
import config

if __name__ == '__main__':

    with open('logging_config.yaml', 'r') as f:
        _conf = yaml.safe_load(f.read())
    logging.config.dictConfig(_conf)
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
    config.refresh_config()

    app = App()
    app.gui()
