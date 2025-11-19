"""
Two layer neural network, without bias

# https://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf

z = x @ W1
h = ReLU(z)
y = h @ W2
L = mean(y^2)

(Untrainable)
    x
    │
    ▼
┌────────┐         ┌──────┐         ┌────────┐         ┌─────┐   ┌──────┐
│ x @ W1 │──→ z ──→│ ReLU │──→ h ──→│ h @ W2 │──→ y ──→│ y^2 │──→│ mean │──→ L (loss)
└────────┘         └──────┘         └────────┘         └─────┘   └──────┘
    ▲                                   ▲
    │                                   │
   W1                                  W2
(Trainable)

Dimensions:
  x:      (batch, input_dim)     e.g., (32, 10)
  y_true: (batch, output_dim)    e.g., (32, 5)

  W1:     (input_dim, hidden)    e.g., (10, 20)
  W2:     (hidden, output_dim)   e.g., (20, 5)

  h:      (batch, hidden)        e.g., (32, 20)
  y:      (batch, output_dim)    e.g., (32, 5)
  loss:   scalar
"""

import torch
import numpy as np
from assertion import assert_loss

# Set seed for reproducibility
np.random.seed(42)

# Set dimensions
batch_size = 32
input_dim = 10
hidden_dim = 20
output_dim = 5
lr = 0.01

# ============================================================================
# Generate Initial Variables (used by both PyTorch and NumPy)
# ============================================================================

# Initialize parameters
W1_initial = np.random.randn(input_dim, hidden_dim).astype(np.float32)
W2_initial = np.random.randn(hidden_dim, output_dim).astype(np.float32)

# Generate synthetic data
x_data = np.random.randn(batch_size, input_dim).astype(np.float32)
y_true_data = np.random.randn(batch_size, output_dim).astype(np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 68.1732,
    20: 10.6691,
    40: 5.2210,
    60: 3.2457,
    80: 2.3180,
}


# ============================================================================
# PyTorch Implementation
# ============================================================================

# Initialize parameters from initial values
W1 = torch.tensor(W1_initial.copy(), dtype=torch.float32, requires_grad=True)
W2 = torch.tensor(W2_initial.copy(), dtype=torch.float32, requires_grad=True)

# Convert data to tensors
x = torch.tensor(x_data, dtype=torch.float32)
y_true = torch.tensor(y_true_data, dtype=torch.float32)

# Create optimizer
optimizer = torch.optim.SGD([W1, W2], lr=lr)

print("=" * 60)
print("PyTorch Two-Layer Network with ReLU")
print("=" * 60)

# Training loop
for epoch in range(100):
    # Forward pass
    z = x @ W1  # [batch_size, hidden_dim]
    h = torch.relu(z)
    y = h @ W2  # [batch_size, output_dim]
    loss = torch.mean(y**2)

    # Backward pass (automatic!)
    loss.backward()

    # To inspect gradients, you can check
    # W1.grad, W2.grad

    # Update all parameters using optimizer
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    if epoch % 20 == 0:
        assert_loss(epoch, loss.item(), epoch_to_expected_loss)
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")

print()

# ============================================================================
# NumPy Implementation
# ============================================================================

# Use the same initial variables as PyTorch
W1 = W1_initial.copy()
W2 = W2_initial.copy()

# Use the same data as PyTorch
x = x_data.copy()
y_true = y_true_data.copy()

print("=" * 60)
print("NumPy Two-Layer Network with ReLU (Manual Backprop)")
print("=" * 60)

# Training loop
for epoch in range(100):
    # ==================== FORWARD PASS ====================
    z = x @ W1  # [batch_size, hidden_dim]
    h = np.maximum(0, z)  # ReLU activation
    y = h @ W2  # [batch_size, output_dim]

    # Compute loss: L = mean(y^2)
    loss = np.mean(y**2)

    # ==================== BACKWARD PASS ====================
    # Loss: L = mean(y^2)
    # dL/dy = 2*y / (batch_size * output_dim)
    dL_dy = (2.0 / (batch_size * output_dim)) * y

    # Second layer: y = h @ W2
    # dL/dW2 = h.T @ dL/dy: [hidden, batch] @ [batch, output] = [hidden, output]
    dL_dW2 = h.T @ dL_dy

    # Backprop to hidden layer: dL/dh = dL/dy @ W2.T
    # dL/dy: [batch, output], W2: [hidden, output]
    # dL/dy @ W2.T: [batch, output] @ [output, hidden] = [batch, hidden]
    dL_dh = dL_dy @ W2.T

    # ReLU derivative: d/dz[ReLU(z)] = 1 if z > 0 else 0
    # Chain rule: dL/dz = dL/dh * (z > 0)
    dL_dz = dL_dh * (z > 0)

    # First layer: z = x @ W1
    # dL/dW1 = x.T @ dL/dz: [input, batch] @ [batch, hidden] = [input, hidden]
    dL_dW1 = x.T @ dL_dz

    # ==================== PARAMETER UPDATE ====================
    W1 -= lr * dL_dW1
    W2 -= lr * dL_dW2

    if epoch % 20 == 0:
        assert_loss(epoch, loss, epoch_to_expected_loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"\nFinal loss: {loss:.4f}")
