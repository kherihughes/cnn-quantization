import unittest
import torch
from src.model.cnn import Net
from src.model.quantized_cnn import QuantizedNet
from src.train import train_model
from src.utils.logging import setup_logger

def load_tests(loader, standard_tests, pattern):
    """Load all test modules."""
    from tests import test_logging, test_cli, test_quantization, test_config
    
    suite = unittest.TestSuite()
    for test_module in (test_logging, test_cli, test_quantization, test_config):
        tests = loader.loadTestsFromModule(test_module)
        suite.addTests(tests)
    return suite

if __name__ == '__main__':
    unittest.main() 