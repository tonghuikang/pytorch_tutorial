"""
Two layer neural network

(Untrainable)
       x                                                         y_true
       │                                                           |
       ▼                                                           ▼
┌─────────────┐   ┌──────┐         ┌─────────────┐         ┌───────────────┐
│ x @ W1 + b1 │──→│ ReLU │──→ h ──→│ h @ W2 + b2 │──→ y ──→│ (y, y_true)^2 │──→ L
└─────────────┘   └──────┘         └─────────────┘         └───────────────┘
       ▲                                  ▲
       │                                  │
     W1,b1                              W2,b2
(Trainable)

Dimensions:
  x:      (batch, input_dim)     e.g., (32, 10)
  W1:     (input_dim, hidden)    e.g., (10, 20)
  b1:     (hidden,)              e.g., (20,)
  h:      (batch, hidden)        e.g., (32, 20)
  W2:     (hidden, output_dim)   e.g., (20, 5)
  b2:     (output_dim,)          e.g., (5,)
  y:      (batch, output_dim)    e.g., (32, 5)
  y_true: (batch, output_dim)    e.g., (32, 5)
"""

import torch
import numpy as np

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
b1_initial = np.random.randn(hidden_dim).astype(np.float32)
W2_initial = np.random.randn(hidden_dim, output_dim).astype(np.float32)
b2_initial = np.random.randn(output_dim).astype(np.float32)

# Generate synthetic data
x_data = np.random.randn(batch_size, input_dim).astype(np.float32)
y_true_data = np.random.randn(batch_size, output_dim).astype(np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 118.0829,
    20: 11.3259,
    40: 5.2974,
    60: 3.3069,
    80: 2.4135,
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
W1 = torch.tensor(W1_initial.copy(), dtype=torch.float32, requires_grad=True)
b1 = torch.tensor(b1_initial.copy(), dtype=torch.float32, requires_grad=True)
W2 = torch.tensor(W2_initial.copy(), dtype=torch.float32, requires_grad=True)
b2 = torch.tensor(b2_initial.copy(), dtype=torch.float32, requires_grad=True)

# Convert data to tensors
x = torch.tensor(x_data, dtype=torch.float32)
y_true = torch.tensor(y_true_data, dtype=torch.float32)

# Create optimizer
optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=lr)

print("=" * 60)
print("PyTorch Two-Layer Network with ReLU")
print("=" * 60)

# Training loop
for epoch in range(100):
    # Forward pass
    h = torch.relu(x @ W1 + b1)
    y = h @ W2 + b2
    loss = torch.mean((y - y_true) ** 2)

    # Backward pass (automatic!)
    loss.backward()

    # To inspect gradients, you can check
    # W1.grad, b1.grad, W2.grad, b2.grad

    # Update all parameters using optimizer
    optimizer.step()

    # Zero gradients
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
W1 = W1_initial.copy()
b1 = b1_initial.copy()
W2 = W2_initial.copy()
b2 = b2_initial.copy()

# Use the same data as PyTorch
x = x_data.copy()
y_true = y_true_data.copy()

print("=" * 60)
print("NumPy Two-Layer Network with ReLU (Manual Backprop)")
print("=" * 60)

# Training loop
for epoch in range(100):
    # ==================== FORWARD PASS ====================
    h = np.maximum(0, x @ W1 + b1)  # ReLU activation
    y = h @ W2 + b2

    # Compute loss
    diff = y - y_true
    loss = np.mean(diff**2)

    # ==================== BACKWARD PASS ====================
    # Gradient of loss with respect to predictions
    # Match PyTorch autograd when loss is mean over all elements
    # dL/d(y) = 2 * (y - y_true) / (batch_size * output_dim)
    dL_dy = (2.0 / (batch_size * output_dim)) * diff

    # Gradient through second layer (W2, b2)
    # Forward: y = h @ W2 + b2  where h is (batch, hidden), W2 is (hidden, output)
    # By chain rule: dL/dW2 = sum over batch of (h[i].T @ dL_dy[i]) for each sample i
    #
    # Why .T? Need to match shapes:
    #   h is (batch, hidden) but we need (hidden, batch) to multiply with dL_dy
    #   dL_dy is (batch, output)
    #   h.T @ dL_dy: (hidden, batch) @ (batch, output) = (hidden, output) matches W2 shape!
    #
    # Intuition: Each weight W2[i,j] affects all batch samples, so we sum contributions
    #            from all samples by doing the matrix multiplication with transposed h
    dL_dW2 = h.T @ dL_dy
    dL_db2 = np.sum(dL_dy, axis=0)

    # Gradient through ReLU activation
    dL_dh = dL_dy @ W2.T
    dL_dh = dL_dh * (h > 0)  # ReLU derivative: 1 if h > 0, else 0

    # Gradient through first layer (W1, b1)
    # Forward: h = x @ W1 + b1  where x is (batch, input), W1 is (input, hidden)
    # By chain rule: dL/dW1 = sum over batch of (x[i].T @ dL_dh[i]) for each sample i
    #
    # Why .T? Need to match shapes:
    #   x is (batch, input) but we need (input, batch) to multiply with dL_dh
    #   dL_dh is (batch, hidden)
    #   x.T @ dL_dh: (input, batch) @ (batch, hidden) = (input, hidden) matches W1 shape!
    dL_dW1 = x.T @ dL_dh
    dL_db1 = np.sum(dL_dh, axis=0)

    # ==================== PARAMETER UPDATE ====================
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

    if epoch % 20 == 0:
        assert_loss(epoch, loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"\nFinal loss: {loss:.4f}")
