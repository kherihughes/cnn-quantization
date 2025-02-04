import unittest
import torch
from src.model.cnn import Net
from src.quantization.weight_quant import quantize_layer_weights

class TestQuantization(unittest.TestCase):
    def test_null_model(self):
        with self.assertRaises(ValueError):
            quantize_layer_weights(None)
            
    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            quantize_layer_weights("not a model")
            
    def test_valid_quantization(self):
        model = Net()
        try:
            quantize_layer_weights(model)
        except Exception as e:
            self.fail(f"Quantization raised unexpected exception: {e}") 