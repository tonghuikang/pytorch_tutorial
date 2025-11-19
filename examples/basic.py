"""
Basic scalar linear model

y = a * x + b
L = mean((y - y_true)^2)

(Untrainable)
      x                     y_true
      │                       |
      ▼                       ▼
┌───────────┐         ┌───────────────┐
│ a * x + b │──→ y ──→│ (y, y_true)^2 │──→ L (loss)
└───────────┘         └───────────────┘
      ▲
      │
     a,b
(Trainable)

Dimensions:

  x:      scalar or [batch, 1]
  y_true: scalar or [batch, 1]

  a:      scalar or [1]
  b:      scalar or [1]

  y:    scalar or [batch, 1]
  loss: scalar

This example demonstrates:
- Simple linear regression
- Gradient computation through a scalar operation
- How PyTorch autograd compares to manual NumPy gradients
"""

import torch
import numpy as np
from assertion import assert_loss

# Set seed for reproducibility
np.random.seed(42)

# Set hyperparameters
lr = 0.01

# ============================================================================
# Generate Initial Variables (used by both PyTorch and NumPy)
# ============================================================================

# Initialize parameters
A_initial = np.float32(np.random.randn())
b_initial = np.float32(np.random.randn())

# Training data
x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y_true_data = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

# Expected loss values are approximate and depend on random initialization
epoch_to_expected_loss = {
    0: 18.007383,
    20: 0.028602,
    40: 0.014697,
    60: 0.013029,
    80: 0.011556,
}


# ============================================================================
# PyTorch Implementation
# ============================================================================

# Initialize parameters from initial values
A = torch.tensor([[A_initial]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([b_initial], dtype=torch.float32, requires_grad=True)

# Convert shared data to tensors
x = torch.tensor(x_data.reshape(-1, 1), dtype=torch.float32)
y_true = torch.tensor(y_true_data.reshape(-1, 1), dtype=torch.float32)

# Create optimizer
optimizer = torch.optim.SGD([A, b], lr=lr)

print("=" * 60)
print("PyTorch Scalar Linear Model")
print("=" * 60)

# Training loop
for epoch in range(100):
    # Forward pass
    y = x @ A.t() + b
    loss = torch.mean((y - y_true) ** 2)

    # Backward pass (automatic!)
    loss.backward()

    # To inspect gradients, you can check
    # A.grad, b.grad

    # Update parameters using optimizer
    optimizer.step()

    # Zero gradients for next iteration
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
A = A_initial.copy()
b = b_initial.copy()

# Use the same data as PyTorch
x = x_data.copy()
y_true = y_true_data.copy()
n = len(x)

print("=" * 60)
print("NumPy Scalar Linear Model (Manual Backprop)")
print("=" * 60)

# Training loop
for epoch in range(100):
    # ==================== FORWARD PASS ====================
    y = A * x + b  # Vectorized computation

    # Compute loss
    diff = y - y_true
    loss = np.mean(diff**2)

    # ==================== BACKWARD PASS ====================
    # We must manually implement the derivatives we derived!

    # dL/dy = (2/n) * (y - y_true)
    dL_dy = (2.0 / n) * diff

    # Chain rule: dL/dA = Sum(dL/dy * dy/dA)
    #                    = Sum(dL_dy * x)
    dL_dA = np.sum(dL_dy * x)

    # Chain rule: dL/db = Sum(dL/dy * dy/db)
    #                    = Sum(dL_dy * 1)
    dL_db = np.sum(dL_dy)

    # ==================== PARAMETER UPDATE ====================
    A -= lr * dL_dA
    b -= lr * dL_db

    if epoch % 20 == 0:
        assert_loss(epoch, loss, epoch_to_expected_loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"Final A: {A:.4f}, b: {b:.4f}")
