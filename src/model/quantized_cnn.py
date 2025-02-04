import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QuantizedNet(nn.Module):
    """Quantized version of the CNN model."""
    
    def __init__(self, original_model: nn.Module):
        """Initialize quantized model from original model."""
        super(QuantizedNet, self).__init__()
        
        # Copy structure and weights from original model
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)
        
        # Copy weights from original model
        with torch.no_grad():
            self.conv1.weight.data = original_model.conv1.weight.data.clone().float()
            self.conv2.weight.data = original_model.conv2.weight.data.clone().float()
            self.fc1.weight.data = original_model.fc1.weight.data.clone().float()
            self.fc2.weight.data = original_model.fc2.weight.data.clone().float()
            self.fc3.weight.data = original_model.fc3.weight.data.clone().float()
        
        # Initialize scaling factors
        self.input_scale: Optional[float] = None
        self.weight_scales = {}
        
        # Activation profiling flag
        self.profile_activations = False
        
        # Set requires_grad to False for all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Initialize weight quantization
        self._quantize_weights()

    def _quantize_weights(self):
        """Quantize model weights and store scaling factors."""
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # Calculate scaling factor using 3-sigma range
                with torch.no_grad():
                    std = layer.weight.data.std()
                    scale = 127 / (3 * std)
                    
                    # Quantize weights but keep as float32
                    quantized = (layer.weight.data * scale).round().clamp(-128, 127) / scale
                    layer.weight.data = quantized
                    self.weight_scales[name] = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization of intermediate activations."""
        # Input quantization
        if self.input_scale is not None:
            x = x * self.input_scale
            x = torch.clamp(x.round(), -128, 127) / self.input_scale
        
        # Conv1 with quantized weights
        x = F.conv2d(x, self.conv1.weight, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.pool(F.relu(x))
        if not self.profile_activations:
            scale = 127 / x.abs().max()
            x = torch.clamp((x * scale).round(), -128, 127) / scale
        
        # Conv2
        x = F.conv2d(x, self.conv2.weight, stride=self.conv2.stride, padding=self.conv2.padding)
        x = self.pool(F.relu(x))
        if not self.profile_activations:
            scale = 127 / x.abs().max()
            x = torch.clamp((x * scale).round(), -128, 127) / scale
        
        # Fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        
        x = F.linear(x, self.fc1.weight)
        x = F.relu(x)
        if not self.profile_activations:
            scale = 127 / x.abs().max()
            x = torch.clamp((x * scale).round(), -128, 127) / scale
        
        x = F.linear(x, self.fc2.weight)
        x = F.relu(x)
        if not self.profile_activations:
            scale = 127 / x.abs().max()
            x = torch.clamp((x * scale).round(), -128, 127) / scale
        
        x = F.linear(x, self.fc3.weight)
        
        return x