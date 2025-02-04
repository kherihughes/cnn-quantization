import torch
from typing import List, Tuple

def quantized_bias(bias: torch.Tensor, n_w: float, n_initial_input: float, 
                  ns: List[Tuple[float, float]]) -> torch.Tensor:
    """Quantize bias values to 32-bit integers.
    
    Args:
        bias (torch.Tensor): Original bias values
        n_w (float): Weight scale factor
        n_initial_input (float): Initial input scale
        ns (List[Tuple[float, float]]): Scale factors from preceding layers
        
    Returns:
        torch.Tensor: Quantized bias values
    """
    # Calculate combined scale
    combined_scale = n_w * n_initial_input
    for weight_scale, output_scale in ns:
        combined_scale *= min(weight_scale, 10.0) * min(output_scale, 10.0)
    
    # Normalize scale to prevent overflow
    combined_scale = combined_scale / (1e6 if combined_scale > 1e6 else 1.0)
    
    # Calculate bias scaling factor
    bias_max = bias.abs().max()
    bias_scale = min(combined_scale, 127 / max(bias_max, 1e-8))
    
    # Quantize and clamp bias
    quantized_bias = torch.clamp((bias * bias_scale).round(), -2147483648, 2147483647)
    
    return quantized_bias

def quantize_model_biases(model: torch.nn.Module) -> None:
    """Quantize all biases in a model.
    
    Args:
        model (nn.Module): Model to quantize
    """
    preceding_scales = []
    for layer in model.children():
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data = quantized_bias(
                layer.bias.data,
                layer.weight.scale,
                model.input_scale,
                preceding_scales[:-1]  # Exclude current layer
            )
            
            # Validation
            if (layer.bias.data < -2147483648).any() or (layer.bias.data > 2147483647).any():
                raise Exception(f"Bias values out of bounds for {layer.__class__.__name__}")
            if (layer.bias.data != layer.bias.data.round()).any():
                raise Exception(f"Non-integer bias values in {layer.__class__.__name__}")