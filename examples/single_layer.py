"""
Single-layer matrix linear model: y = x @ A + b

(Untrainable)
      x                     y_true
      │                       |
      ▼                       ▼
┌───────────┐         ┌───────────────┐
│ A * x + b │──→ y ──→│ (y, y_true)^2 │──→ L (loss)
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
- Batch processing
- Gradient computation through matrix operations
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
    0: 4.0978,
    20: 2.5171,
    40: 1.6226,
    60: 1.1016,
    80: 0.7871,
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
    y = x @ A + b  # [batch_size, output_dim]
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
    # y = x @ A + b (broadcast b)
    y = x @ A + b  # Shape: [batch_size, output_dim]

    # Compute loss
    diff = y - y_true
    loss = np.mean(diff**2)

    # ==================== BACKWARD PASS ====================
    # Match PyTorch autograd when loss is mean over all elements
    # dL/d(y) = 2 * (y - y_true) / (batch_size * output_dim)
    dL_dy = (2.0 / (batch_size * output_dim)) * diff

    # dL/dA = x.T @ dL_dy
    dL_dA = x.T @ dL_dy

    # dL/db = sum(dL_dy, axis=0)
    dL_db = np.sum(dL_dy, axis=0)

    # ==================== PARAMETER UPDATE ====================
    A -= lr * dL_dA
    b -= lr * dL_db

    if epoch % 20 == 0:
        assert_loss(epoch, loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"\nFinal loss: {loss:.4f}")
