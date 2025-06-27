import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger that writes to both a file and the console.

    Args:
        name (str): The name of the logger.
        log_file (Path): The path to the log file.
        level (int): The logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger