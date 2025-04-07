# -------------------------------------------------------------------------------- #
# Logger
# -------------------------------------------------------------------------------- #
"""
Logger module for Astral AI.

Usage Instructions
------------------
1. Install Dependencies:
   - pip install python-dotenv colorama

2. Environment Variables (optional):
   - VERBOSE:
       - Set to "true", "1", or "yes" to enable DEBUG (verbose) logging.
       - Default: "false" (INFO-level logging).
   - LOG_TO_FILE:
       - Set to "true", "1", or "yes" to enable file logging.
       - Default: "false" (disabled).
   - LOG_DIRECTORY:
       - Directory path for log files. Used only if LOG_TO_FILE is true.
       - Default: "logs"
   - LOG_FILENAME:
       - Filename for the log file. Used only if LOG_TO_FILE is true.
       - Default: "app.log"

3. How to Set Up .env File:
   Example .env:
       VERBOSE=true
       LOG_TO_FILE=true
       LOG_DIRECTORY=my_custom_logs
       LOG_FILENAME=my_app.log

4. Usage in Code:
   from logger import logger

   # Basic usage
   logger.info("Information message.")
   logger.debug("Debug message (shown if VERBOSE=true).")

   # If you want to override .env programmatically:
   custom_logger = Logger(
       verbose=True, 
       log_to_file=True, 
       log_directory="custom_dir", 
       log_filename="mylog.log"
   )
   custom_logger.info("This goes to both console (colored) and custom_dir/mylog.log")

General Behavior
----------------
- By default, the logger only prints to console (colored) with level INFO or DEBUG 
  (if VERBOSE=true). 
- If you explicitly enable file logging (log_to_file=True or LOG_TO_FILE=true in .env),
  a rotating file handler is added to store logs (in plain text).

"""
# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for cross-platform compatibility
init(autoreset=True)

# Load environment variables from .env
load_dotenv(dotenv_path=".env", override=True)

# -------------------------------------------------------------------------------- #
# Custom TRACE Level
# -------------------------------------------------------------------------------- #

TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = trace

# -------------------------------------------------------------------------------- #
# Logger Configuration
# -------------------------------------------------------------------------------- #

class Logger:
    """
    A configurable, singleton logger with optional colored console output and 
    optional file logging (disabled by default). Log level is determined by:
      1) Constructor arguments, which override
      2) Environment variables, which default to: VERBOSE=false => INFO

    Parameters
    ----------
    name : str
        Name of the logger. Defaults to "Astral Logger".
    verbose : Optional[bool]
        If True, sets console handler level to DEBUG (unless trace_logging is True).
        If False, sets to INFO (unless trace_logging is True).
        If None, checks VERBOSE env var.
    trace_logging : Optional[bool]
        If True, sets console handler level to TRACE (level 5), overriding verbose.
        If None, checks ASTRAL_TRACE_LOGGING env var.
        Defaults to False. Undocumented for end-users.
    log_to_file : Optional[bool]
        If True, enable file logging. If None, check LOG_TO_FILE env var.
    log_directory : str
        The folder path for log files. Only used if file logging is enabled.
        Defaults to "logs" or LOG_DIRECTORY env var.
    log_filename : str
        The log filename. Only used if file logging is enabled.
        Defaults to "app.log" or LOG_FILENAME env var.
    log_format : str
        The logging format string.
    date_format : str
        The logging date/time format string.

    Singleton Behavior
    ------------------
    If you create multiple `Logger` instances, they all share the 
    same underlying logger object and handlers, preventing duplicate logs.

    Methods
    -------
    trace(msg, *args, **kwargs)
        Log a TRACE-level message (level 5).
    info(msg, *args, **kwargs)
        Log an INFO-level message.
    debug(msg, *args, **kwargs)
        Log a DEBUG-level message.
    warning(msg, *args, **kwargs)
        Log a WARNING-level message.
    error(msg, *args, **kwargs)
        Log an ERROR-level message.
    critical(msg, *args, **kwargs)
        Log a CRITICAL-level message.
    """

    _instance = None  # Singleton instance

    COLOR_MAP = {
        "TRACE": Fore.MAGENTA + Style.DIM,  # Added color for TRACE
        "DEBUG": Fore.BLUE + Style.BRIGHT,
        "INFO": Fore.GREEN + Style.BRIGHT,
        "WARNING": Fore.YELLOW + Style.BRIGHT,
        "ERROR": Fore.RED + Style.BRIGHT,
        "CRITICAL": Fore.RED + Style.BRIGHT
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str = "Astral Logger",
        verbose: Optional[bool] = None,
        trace_logging: Optional[bool] = None,
        log_to_file: Optional[bool] = None,
        log_directory: Optional[str] = None,
        log_filename: Optional[str] = None,
        log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        # If logger is already initialized, skip reconfiguration
        if hasattr(self, "logger"):
            return

        # Determine trace logging first
        if trace_logging is None:
            env_trace = os.getenv("ASTRAL_TRACE_LOGGING", "false").strip().lower()
            trace_logging = env_trace in ("true", "1", "yes")

        # Determine console verbosity based on trace_logging, then verbose/env
        if trace_logging:
            console_level = TRACE_LEVEL_NUM
        else:
            if verbose is None:
                env_verbose = os.getenv("VERBOSE", "false").strip().lower()
                verbose = env_verbose in ("true", "1", "yes")
            console_level = logging.DEBUG if verbose else logging.INFO

        # Decide file logging
        if log_to_file is None:
            env_log_to_file = os.getenv("LOG_TO_FILE", "false").strip().lower()
            log_to_file = env_log_to_file in ("true", "1", "yes")

        # Determine log directory and filename
        log_directory = log_directory or os.getenv("LOG_DIRECTORY", "logs")
        log_filename = log_filename or os.getenv("LOG_FILENAME", "app.log")

        # Initialize the core logger - set its base level to the lowest we might use (TRACE)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(TRACE_LEVEL_NUM) # Allow TRACE messages through the logger itself

        # Console handler (colored output) - level controlled by trace_logging then verbose
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(console_level) # Set level based on priority
        stdout_handler.setFormatter(self.ColoredFormatter(log_format, date_format))
        self.logger.addHandler(stdout_handler)

        # Optional file handler - logs everything from TRACE level up if enabled
        if log_to_file:
            log_path = Path(log_directory)
            log_path.mkdir(parents=True, exist_ok=True)
            file_formatter = logging.Formatter(log_format, date_format)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / log_filename,
                mode="a",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            file_handler.setLevel(TRACE_LEVEL_NUM) # Log TRACE+ to file
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # If no handlers are attached (edge case), add a NullHandler
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def trace(self, msg: str, *args, **kwargs):
        """Log a message with severity 'TRACE' (level 5)."""
        self.logger.trace(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log a message with severity 'INFO'."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log a message with severity 'DEBUG'."""
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log a message with severity 'WARNING'."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log a message with severity 'ERROR'."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log a message with severity 'CRITICAL'."""
        self.logger.critical(msg, *args, **kwargs)

    class ColoredFormatter(logging.Formatter):
        """
        Custom formatter to add colors to log levels for terminal output only.
        Restores the original levelname so it does not affect other handlers 
        (like file logging).
        """

        def format(self, record):
            original_levelname = record.levelname
            color = Logger.COLOR_MAP.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            try:
                return super().format(record)
            finally:
                # Restore the original levelname so file logs remain colorless
                record.levelname = original_levelname


# -------------------------------------------------------------------------------- #
# Default Logger Instance
# -------------------------------------------------------------------------------- #

logger = Logger()