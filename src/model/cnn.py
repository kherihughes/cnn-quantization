import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Basic CNN model for CIFAR-10 classification."""
    
    def __init__(self, device='cuda'):
        """Initialize the CNN model.
        
        Args:
            device (str): Device to place model on
        """
        super(Net, self).__init__()
        self.device = device
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)
        
        # Initialize scale attribute
        self.input_scale = None
        self.profile_activations = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, 10)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x