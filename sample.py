"""Simple neural network example demonstrating forward and backward passes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameters
input_size: int = 3072  # 32x32x3 (e.g., CIFAR-10 flattened)
hidden_size: int = 128
output_size: int = 10  # 10 classes
learning_rate: float = 0.001
batch_size: int = 64
num_epochs: int = 3

# Create model and optimizer
model: SimpleNN = SimpleNN(input_size, hidden_size, output_size)
optimizer: AdamW = AdamW(model.parameters(), lr=learning_rate)

# Dummy data (replace with real dataset)
num_samples: int = 1000
X_train: Tensor = torch.randn(num_samples, input_size)
y_train: Tensor = torch.randint(0, output_size, (num_samples,))

# Create DataLoader
train_dataset: TensorDataset = TensorDataset(X_train, y_train)
train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    total_loss: float = 0.0
    for inputs, labels in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions: Tensor = model(inputs)

        # Compute loss
        loss: Tensor = F.cross_entropy(predictions, labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    avg_loss: float = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

print("Training complete!")
