import logging
import sys


_DEFAULT_LOGGER = logging.getLogger()
_DEFAULT_FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    _DEFAULT_LOGGER.handlers = []
    _DEFAULT_LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_DEFAULT_FORMATTER)
        handler.setLevel(stderr_level)
        _DEFAULT_LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(_DEFAULT_FORMATTER)
        handler.setLevel(file_level)
        _DEFAULT_LOGGER.addHandler(handler)

    _DEFAULT_LOGGER.info('logger set up')
    return _DEFAULT_LOGGER
