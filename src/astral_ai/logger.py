# -------------------------------------------------------------------------------- #
# Logger
# -------------------------------------------------------------------------------- #

"""
This module contains the logger for the Astral AI project.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in imports
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Logger imports
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for cross-platform compatibility
init(autoreset=True)

# Load environment variables from .env
load_dotenv()

# -------------------------------------------------------------------------------- #
# Logger Configuration
# -------------------------------------------------------------------------------- #


class Logger:
    """
    Singleton Logger with configurable outputs, formatting, file handling, 
    and colored log levels for terminal output only (no color in file logs).
    """

    _instance = None  # Singleton instance

    COLOR_MAP = {
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
        name: str = "AI Logger",
        log_level: Optional[str] = None,
        log_directory: str = "logs",
        log_filename: str = "app.log",
        log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        # Skip reconfiguration if already initialized
        if hasattr(self, "logger"):
            return

        # Create the logs directory if it doesn't exist
        log_path = Path(log_directory)
        log_path.mkdir(parents=True, exist_ok=True)

        # Set log level from .env or default to INFO
        level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()

        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level, logging.INFO))

        # Plain text formatter for file handler
        file_formatter = logging.Formatter(log_format, date_format)

        # Configure rotating file handler (no colors)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / log_filename,
            mode="w",  # Overwrites the log file on every run
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(self.logger.level)
        file_handler.setFormatter(file_formatter)

        # Colored formatter for stream handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(self.logger.level)
        stdout_handler.setFormatter(self.ColoredFormatter(log_format, date_format))

        # Add handlers if not already present
        if not self.logger.handlers:
            self.logger.addHandler(stdout_handler)  # Terminal (colored)
            self.logger.addHandler(file_handler)     # File (plain text)

    def info(self, msg: str, *args, **kwargs):
        """Pass-through for self.logger.info()."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Pass-through for self.logger.debug()."""
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Pass-through for self.logger.warning()."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Pass-through for self.logger.error()."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Pass-through for self.logger.critical()."""
        self.logger.critical(msg, *args, **kwargs)

    class ColoredFormatter(logging.Formatter):
        """
        Custom formatter to add colors to log levels for terminal output only.
        Restores the original levelname so it doesn't affect other handlers.
        """

        def format(self, record):
            original_levelname = record.levelname
            color = Logger.COLOR_MAP.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            try:
                return super().format(record)
            finally:
                # Restore the original levelname so other handlers
                # (e.g., file handlers) are not affected.
                record.levelname = original_levelname


# -------------------------------------------------------------------------------- #
# Create default logger instance (demo)
# -------------------------------------------------------------------------------- #

logger = Logger()
