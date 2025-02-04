import logging
import sys
from typing import Optional

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Configure and return a logger with consistent formatting.
    
    Args:
        name (str): Name of the logger
        level (int, optional): Logging level. Defaults to INFO
        
    Returns:
        logging.Logger: Configured logger
        
    Examples:
        >>> logger = setup_logger("test")
        >>> isinstance(logger, logging.Logger)
        True
        >>> logger.name
        'test'
    """
    if level is None:
        level = logging.INFO
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler if not already added
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger 