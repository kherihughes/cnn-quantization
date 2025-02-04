import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

def train_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,  # Force remaining arguments to be keyword-only
    epochs: int = 2,
    device: str = 'cuda'
) -> torch.nn.Module:
    """Train a model on CIFAR-10 dataset.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        epochs: Number of training epochs (keyword-only)
        device: Device to train on (keyword-only)
        
    Returns:
        Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return model

def test_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
               max_samples: Optional[int] = None, device: str = 'cuda') -> float:
    """Test a model and return accuracy.
    
    Args:
        model: Model to test
        dataloader: Test data loader
        max_samples: Maximum number of samples to test (optional)
        device: Device to run test on
        
    Returns:
        float: Accuracy percentage
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if max_samples and total >= max_samples:
                break
                
    return 100 * correct / total 