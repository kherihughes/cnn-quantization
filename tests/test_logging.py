import unittest
import logging
from src.utils.logging import setup_logger

class TestLogging(unittest.TestCase):
    def test_logger_creation(self):
        logger = setup_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.INFO)
        
    def test_logger_custom_level(self):
        logger = setup_logger("test_logger", logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)
        
    def test_logger_handler(self):
        logger = setup_logger("test_logger")
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler) 