import torch
import numpy as np
from typing import List, Tuple

def quantize_activations(activations: np.ndarray, n_w: float, n_initial_input: float, 
                        ns: List[Tuple[float, float]]) -> float:
    """Calculate scaling factor for layer activations.
    
    Args:
        activations (np.ndarray): Layer activation values
        n_w (float): Weight scaling factor
        n_initial_input (float): Initial input scaling factor
        ns (List[Tuple[float, float]]): Weight and output scales for preceding layers
        
    Returns:
        float: Activation scaling factor
    """
    std = activations.std()
    scale = 127 / (4 * std)  # Use 4-sigma range for activations
    
    # Dynamically adjust scale to ensure range coverage
    max_activation = activations.max()
    min_activation = activations.min()
    adjusted_scale = 127 / max(abs(max_activation), abs(min_activation))
    scale = min(scale, adjusted_scale)
    
    return scale

def register_activation_profiling_hooks(model: torch.nn.Module) -> None:
    """Register hooks to collect activation statistics during forward pass.
    
    Args:
        model (nn.Module): Model to profile
    """
    # Initialize empty arrays for activation storage
    model.input_activations = []
    
    def input_hook(module, input_tensor, output):
        if model.profile_activations:
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            model.input_activations.append(input_tensor.detach().cpu())
            
            # Calculate input scale after collecting enough samples
            if len(model.input_activations) >= 10:  # Adjust this number as needed
                input_data = torch.cat(model.input_activations)
                std = input_data.std()
                max_val = input_data.abs().max()
                
                # Use the more conservative of 4-sigma or max value scaling
                scale_sigma = 127 / (4 * std)
                scale_max = 127 / max_val
                model.input_scale = min(scale_sigma, scale_max)
                
                # Clear stored activations to save memory
                model.input_activations = []

    # Register hook on first layer
    model.conv1.register_forward_hook(input_hook)

def apply_activation_quantization(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply quantization to activations.
    
    Args:
        x (torch.Tensor): Input activations
        scale (float): Scaling factor
        
    Returns:
        torch.Tensor: Quantized activations
    """
    x = x * scale
    x = torch.clamp(x.round(), -128, 127)
    return x