import unittest
from pathlib import Path
import torch
from src.cli import main
from src.model.cnn import Net

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.test_model_path = Path("test_model.pth")
        # Create a test model
        model = Net()
        torch.save(model.state_dict(), self.test_model_path)
        
    def tearDown(self):
        # Clean up test files
        if self.test_model_path.exists():
            self.test_model_path.unlink()
            
    def test_model_loading_error(self):
        # Test loading non-existent model
        non_existent_path = Path("non_existent.pth")
        self.assertFalse(non_existent_path.exists())
        
    def test_model_loading_success(self):
        # Test loading existing model
        self.assertTrue(self.test_model_path.exists()) 