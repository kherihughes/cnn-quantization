# CNN Post-Training Quantization

This project implements post-training quantization techniques for a CNN model trained on the CIFAR-10 dataset. The goal is to demonstrate how to reduce model precision while maintaining accuracy, enabling efficient deployment on resource-constrained hardware.

## Overview

Post-training quantization reduces model precision from 32-bit floating-point to:
- 8-bit integers for weights and activations 
- 32-bit integers for biases

The project implements systematic quantization techniques including:
- Weight quantization using 3-sigma scaling
- Activation quantization with dynamic range adjustment
- Bias quantization with combined layer scaling

## Project Structure

```
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cnn.py           # Base CNN architecture
│   │   └── quantized_cnn.py # Quantized model implementation
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── weight_quant.py  # Weight quantization functions
│   │   ├── act_quant.py     # Activation quantization functions
│   │   └── bias_quant.py    # Bias quantization functions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py       # Logging configuration
│   │   └── visualization.py  # Plotting utilities
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration parameters
│   └── train.py            # Training and testing functions
├── tests/                  # Unit tests
│   ├── test_cli.py
│   ├── test_components.py
│   ├── test_config.py
│   ├── test_logging.py
│   └── test_quantization.py
├── notebooks/
│   └── quantization_demo.ipynb # End-to-end demonstration
└── requirements.txt
```

## Results

The quantization pipeline maintains model accuracy while significantly reducing precision:

| Stage | Test Accuracy (%) |
|-------|------------------|
| Before Quantization | 53.44 |
| After Weight Quantization | 54.10 |
| After Full Quantization | 54.10 |

## Requirements

```bash
# For CUDA 12.x support
pip install -r requirements.txt
```

Required packages:
- PyTorch 2.2.0+cu121
- torchvision 0.17.0+cu121
- NumPy ≥1.21.0
- Matplotlib ≥3.3.0

## Usage

### Training and Quantization

```python
from src.model.cnn import Net
from src.model.quantized_cnn import QuantizedNet
from src.quantization.weight_quant import quantize_layer_weights
from src.quantization.act_quant import register_activation_profiling_hooks
from src.quantization.bias_quant import quantize_model_biases
from src.train import train_model, test_model

# Train base model
net = Net()
train_model(net, trainloader, epochs=2)
base_accuracy = test_model(net, testloader)

# Create and profile quantized model
quantized_net = QuantizedNet(net)
quantized_net.profile_activations = True
register_activation_profiling_hooks(quantized_net)

# Quantize weights and test
quantize_layer_weights(quantized_net)
accuracy = test_model(quantized_net, testloader)

# Quantize biases and final test
quantize_model_biases(quantized_net)
final_accuracy = test_model(quantized_net, testloader)
```

### Visualization Tools

```python
from src.utils.visualization import plot_weight_distribution, plot_activation_distribution

# Visualize weight distributions
layers = [net.conv1, net.conv2, net.fc1, net.fc2, net.fc3]
layer_names = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']

for layer, name in zip(layers, layer_names):
    plot_weight_distribution(layer.weight.data, name)

# Visualize activations
plot_activation_distribution(activations, name)