"""Configuration parameters for model training and quantization.

This module contains all configurable parameters used throughout the project,
including training hyperparameters, model architecture settings, and 
quantization options.
"""

import torch

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 2

# Model parameters  
NUM_CLASSES = 10
HIDDEN_SIZE = 512

# Quantization parameters
WEIGHT_BITS = 8
ACTIVATION_BITS = 8
BIAS_BITS = 8

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
MODEL_DIR = 'models'
LOG_DIR = 'logs' 