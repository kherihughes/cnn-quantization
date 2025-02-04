import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_weight_distribution(weights: torch.Tensor, layer_name: str) -> None:
    """Plot histogram of weight values for a layer.
    
    Args:
        weights (torch.Tensor): Weight tensor
        layer_name (str): Name of the layer
    """
    # Flatten weights to 1D for histogram
    weights_flat = weights.cpu().flatten().numpy()
    
    plt.figure()
    plt.hist(weights_flat, bins=50)
    plt.title(f'Weight distribution of {layer_name}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.show()

def plot_activation_distribution(activations: np.ndarray, name: str) -> None:
    """Plot histogram of activation values.
    
    Args:
        activations (np.ndarray): Activation values
        name (str): Name of the layer
    """
    plt.figure()
    plt.hist(activations, bins=50)
    plt.title(f'Activation distribution of {name}')
    plt.xlabel('Activation value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Print statistics
    activation_range = activations.max() - activations.min()
    activation_mean = activations.mean()
    activation_std = activations.std()
    activation_3sigma_range = (
        activation_mean - 3 * activation_std,
        activation_mean + 3 * activation_std
    )
    
    print(f"{name} range: {activation_range:.6f}")
    print(f"{name} 3-sigma range: ({activation_3sigma_range[0]:.6f}, {activation_3sigma_range[1]:.6f})")

def plot_accuracy_comparison(accuracies: dict) -> None:
    """Plot accuracy comparison between different quantization stages.
    
    Args:
        accuracies (dict): Dictionary of accuracies for each stage
    """
    stages = list(accuracies.keys())
    values = list(accuracies.values())
    
    # Plot directly on current axes
    plt.bar(stages, values)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Set y-axis from 0 to 100 for percentages 