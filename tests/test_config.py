import unittest
from src import config

class TestConfig(unittest.TestCase):
    def test_batch_size_positive(self):
        self.assertGreater(config.BATCH_SIZE, 0)
        
    def test_learning_rate_range(self):
        self.assertGreater(config.LEARNING_RATE, 0)
        self.assertLess(config.LEARNING_RATE, 1)
        
    def test_quantization_bits(self):
        self.assertEqual(config.WEIGHT_BITS, 8)
        self.assertEqual(config.ACTIVATION_BITS, 8)
        self.assertEqual(config.BIAS_BITS, 8) 