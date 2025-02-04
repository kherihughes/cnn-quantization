import torch
from typing import Tuple

def quantized_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize weights to 8-bit integers using 3-sigma scaling."""
    # Ensure weights are in float32 for statistics
    weights_float = weights.float()
    
    # Calculate scaling factor using 3-sigma range
    std = weights_float.std()
    scale = 127 / (3 * std)
    
    # Quantize weights
    quantized = (weights_float * scale).round()
    
    # Clamp to 8-bit signed integer range
    quantized = torch.clamp(quantized, min=-128, max=127)
    
    return quantized, scale

def quantize_layer_weights(model: torch.nn.Module) -> None:
    """Quantize weights to 8-bit integers."""
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            # Ensure we're working with float32 for calculations
            with torch.no_grad():
                weights_float = layer.weight.data.float()
                
                # Calculate scaling factor using 3-sigma range
                std = weights_float.std()
                scale = 127 / (3 * std)
                
                # Quantize weights to int8 range but keep as float32
                quantized = (weights_float * scale).round().clamp(-128, 127) / scale
                
                # Store quantized weights and scale
                layer.weight.data = quantized
                layer.weight.scale = scale