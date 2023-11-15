# -*- coding: utf-8 -*-

import logging
import os
import sys

LEVEL = logging.INFO
FILE_NAME = "log"

log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class Logger(object):

    def __init__(self, name, level=LEVEL, filename=FILE_NAME):
        logger = logging.getLogger(name)
        logger.setLevel(LEVEL)

        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        templogfile = os.path.join(log_dir, filename)

        file_handler = logging.FileHandler(templogfile)
        file_handler.setLevel(LEVEL)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logger

    def setLevel(self, level):
        self.logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
