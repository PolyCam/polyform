import os
import logging

"""
Helper functions to make it faster to work with Python's standard logging library
"""

def set_log_level(logger, default_level=logging.INFO):
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        exec("logger.setLevel(logging.{})".format(level))
    else:
        logger.setLevel(default_level)


def setup_logger(name):
    logger = logging.getLogger(name)
    set_log_level(logger)
    logging.basicConfig()
    return logger


def class_logger(obj):
    """ initializes a logger with type name of obj """
    return setup_logger(type(obj).__name__)


def file_logger(file):
    this_file = os.path.splitext(os.path.basename(file))[0]
    logger = setup_logger(' ' + this_file + ' ')
    return logger


# A common logger instance that can be used when a dedicated
# named logger is not needed
logger = setup_logger('polyform')