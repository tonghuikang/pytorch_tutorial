"""
One layer neural network

z = x @ A + b
y = ReLU(z)
L = mean((y - y_true)^2)

(Untrainable)
      x                     y_true
      │                       |
      ▼                       ▼
┌───────────┐         ┌───────────────┐
│ x @ A + b │──→ y ──→│ (y, y_true)^2 │──→ L (loss)
└───────────┘         └───────────────┘
      ▲
      │
     A,b
(Trainable)

Dimensions:
  x:      (batch_size, input_dim)     e.g., (4, 5)
  y_true: (batch_size, output_dim)    e.g., (4, 3)

  A:      (input_dim, output_dim)     e.g., (5, 3)
  b:      (output_dim,)               e.g., (3,)

  y:      (batch_size, output_dim)    e.g., (4, 3)
  loss:   scalar

This example demonstrates:
- Matrix multiplication and broadcasting
- Activation functions (ReLU)
- Batch processing
- Gradient computation through matrix operations and activation functions
"""

import torch
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Set dimensions
batch_size = 4
input_dim = 5
output_dim = 3
lr = 0.01

# ============================================================================
# Generate Initial Variables (used by both PyTorch and NumPy)
# ============================================================================

# Initialize parameters
A_initial = np.random.randn(input_dim, output_dim).astype(np.float32)
b_initial = np.random.randn(output_dim).astype(np.float32)

# Training data
x_data = np.random.randn(batch_size, input_dim).astype(np.float32)
y_true_data = np.random.randn(batch_size, output_dim).astype(np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 2.8235,
    20: 1.8644,
    40: 1.3089,
    60: 0.9868,
    80: 0.8215,
}


def assert_loss(epoch: int, loss: float) -> None:
    if epoch in epoch_to_expected_loss:
        assert abs(loss - epoch_to_expected_loss[epoch]) < 0.001, (
            f"Epoch {epoch}: expected {epoch_to_expected_loss[epoch]}, got {loss}"
        )


# ============================================================================
# PyTorch Implementation
# ============================================================================

# Initialize parameters from initial values
A = torch.tensor(A_initial.copy(), dtype=torch.float32, requires_grad=True)
b = torch.tensor(b_initial.copy(), dtype=torch.float32, requires_grad=True)

# Convert data to tensors
x = torch.tensor(x_data, dtype=torch.float32)
y_true = torch.tensor(y_true_data, dtype=torch.float32)

# Create optimizer
optimizer = torch.optim.SGD([A, b], lr=lr)

print("=" * 60)
print("PyTorch Single-Layer Matrix Model")
print("=" * 60)

# Training loop
for epoch in range(100):
    z = x @ A + b  # [batch_size, output_dim]
    y = torch.relu(z)  # Apply ReLU activation
    loss = torch.mean((y - y_true) ** 2)

    loss.backward()

    # To inspect gradients, you can check
    # A.grad, b.grad

    # Update parameters using optimizer
    optimizer.step()

    # Zero gradients for next iteration
    optimizer.zero_grad()

    if epoch % 20 == 0:
        assert_loss(epoch, loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")

print()

# ============================================================================
# NumPy Implementation
# ============================================================================

# Use the same initial variables as PyTorch
A = A_initial.copy()
b = b_initial.copy()

# Use the same data as PyTorch
x = x_data.copy()
y_true = y_true_data.copy()

print("=" * 60)
print("NumPy Single-Layer Matrix Model (Manual Backprop)")
print("=" * 60)

# Training loop
for epoch in range(100):
    # ==================== FORWARD PASS ====================
    # z = x @ A + b (broadcast b)
    z = x @ A + b  # Shape: [batch_size, output_dim]
    y = np.maximum(0, z)  # Apply ReLU activation

    # Compute loss
    diff = y - y_true
    loss = np.mean(diff**2)

    # ==================== BACKWARD PASS ====================
    # Loss function: L = mean((y - y_true)^2)
    # dL/dy = d/dy[mean((y - y_true)^2)] = 2(y - y_true) / (batch_size * output_dim)
    # Dividing by (batch_size * output_dim) because mean reduces over all elements
    dL_dy = (2.0 / (batch_size * output_dim)) * diff

    # ReLU derivative: d/dz[ReLU(z)] = 1 if z > 0 else 0
    # Chain rule: dL/dz = dL/dy * dy/dz = dL/dy * (z > 0)
    dL_dz = dL_dy * (z > 0)

    # For linear layer: z = x @ A + b, shape [batch_size, output_dim]
    # dL/dA = d/dA[L] where each element A[i,j] affects z[:,j] through x[:,i]
    # By chain rule: dL/dA = x.T @ dL/dz
    # x has shape [batch_size, input_dim], dL/dz has shape [batch_size, output_dim]
    # x.T has shape [input_dim, batch_size], so x.T @ dL/dz gives [input_dim, output_dim]
    dL_dA = x.T @ dL_dz

    # For bias: b[j] adds directly to z[:,j]
    # dL/db[j] = sum_over_batch(dL/dz[:,j])
    # Sum over axis=0 (batch dimension) to get [output_dim,]
    dL_db = np.sum(dL_dz, axis=0)

    # ==================== PARAMETER UPDATE ====================
    A -= lr * dL_dA
    b -= lr * dL_db

    if epoch % 20 == 0:
        assert_loss(epoch, loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"\nFinal loss: {loss:.4f}")
