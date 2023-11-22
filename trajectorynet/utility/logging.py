import os
import sys
from logging import DEBUG, INFO, Formatter, Logger, StreamHandler, FileHandler, getLogger
from typing import Optional

# get DEBUG environment variable
DEBUG_ENV = os.environ.get("DEBUG")

def init_logger(name: Optional[str]=None, filename: Optional[str]=None)-> Logger:
    """Initialize logger.

    Parameters
    ----------
    name : Optional[str], optional
        Name of logger, by default None
    filename : Optional[str], optional
        Name of log file, by default None

    Returns
    -------
    Logger
        Logger instance
    """
    _logger = getLogger(__name__ if name is None else name)
    _logger.setLevel(DEBUG if DEBUG_ENV else INFO)
    _handler = FileHandler(filename=filename) if filename is not None else StreamHandler(sys.stdout)
    _handler.setLevel(DEBUG if DEBUG_ENV else INFO)
    _formatter = Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    return _logger