"""
Logistic regression with cross-entropy loss for binary classification.

(Untrainable)
      x                                                  y_true
      │                                                    |
      ▼                                                    ▼
┌───────────┐              ┌─────────┐             ┌───────────────┐
│ x @ W + b │──→ logits ──→│ softmax │──→ probs ──→│ cross_entropy │──→ L (loss)
└───────────┘              └─────────┘             └───────────────┘
      ▲
      │
     W,b
(Trainable)

Dimensions:
  x:      (batch_size, input_dim)   e.g., (16, 5)
  y_true: (batch_size,)             e.g., (16,)

  W:      (input_dim, num_classes)  e.g., (5, 2)
  b:      (num_classes,)            e.g., (2,)

  logits: (batch_size, num_classes) e.g., (16, 2)
  probs:  (batch_size, num_classes) e.g., (16, 2)
  loss:   scalar

This example demonstrates:
- Classification vs regression
- Cross-entropy loss
- Softmax activation
- How proper loss functions affect gradient flow
"""

import torch
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Set dimensions
batch_size = 16
input_dim = 5
num_classes = 2
lr = 0.01

# ============================================================================
# Generate Initial Variables (used by both PyTorch and NumPy)
# ============================================================================

# Initialize parameters
W_initial = np.random.randn(input_dim, num_classes).astype(np.float32)
b_initial = np.random.randn(num_classes).astype(np.float32)

# Generate synthetic data
x_data = np.random.randn(batch_size, input_dim).astype(np.float32)
y_true_data = np.random.randint(0, num_classes, batch_size)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 1.0743,
    20: 0.9850,
    40: 0.9077,
    60: 0.8411,
    80: 0.7836,
}


def assert_loss(epoch: int, loss: float) -> None:
    if epoch in epoch_to_expected_loss:
        assert abs(loss - epoch_to_expected_loss[epoch]) < 0.001, (
            f"Epoch {epoch}: expected {epoch_to_expected_loss[epoch]}, got {loss}"
        )


# ============================================================================
# PyTorch Implementation with Cross-Entropy Loss
# ============================================================================

# Initialize parameters from initial values
W = torch.tensor(W_initial.copy(), dtype=torch.float32, requires_grad=True)
b = torch.tensor(b_initial.copy(), dtype=torch.float32, requires_grad=True)

# Convert data to tensors
x = torch.tensor(x_data, dtype=torch.float32)
y_true = torch.tensor(y_true_data, dtype=torch.long)

# Create optimizer
optimizer = torch.optim.SGD([W, b], lr=lr)

# Cross-entropy loss (includes softmax internally)
criterion = torch.nn.CrossEntropyLoss()

print("=" * 60)
print("PyTorch Binary Classification (Cross-Entropy Loss)")
print("=" * 60)

# Training loop
for epoch in range(100):
    # Forward pass
    logits = x @ W + b  # [batch_size, num_classes]
    loss = criterion(logits, y_true)

    # Backward pass (automatic!)
    loss.backward()

    # Update parameters
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    if epoch % 20 == 0:
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.mean((predictions == y_true).float()).item()
        assert_loss(epoch, loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

print()

# ============================================================================
# NumPy Implementation with Manual Cross-Entropy Gradient
# ============================================================================

# Use the same initial variables as PyTorch
W = W_initial.copy()
b = b_initial.copy()

# Use the same data as PyTorch
x = x_data.copy()
y_true = y_true_data.copy()

print("=" * 60)
print("NumPy Binary Classification (Manual Cross-Entropy)")
print("=" * 60)


# Helper function: softmax
def softmax(z):
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Training loop
for epoch in range(100):
    # ==================== FORWARD PASS ====================
    logits = x @ W + b  # [batch_size, num_classes]
    probs = softmax(logits)

    # Cross-entropy loss: -sum(y_true * log(probs)) / batch_size
    # For one-hot encoded y_true, this is: -log(probs[i, y_true[i]]) / batch_size
    loss = -np.mean(np.log(probs[np.arange(batch_size), y_true]))

    # ==================== BACKWARD PASS ====================
    # For cross-entropy + softmax, the gradient simplifies to:
    # dL/dlogits[i, j] = probs[i, j] - y_true_onehot[i, j]
    # This is one of the most elegant results in deep learning!

    # Create one-hot encoding
    y_true_onehot = np.zeros((batch_size, num_classes))
    y_true_onehot[np.arange(batch_size), y_true] = 1

    # Gradient of loss w.r.t. logits
    dL_dlogits = (probs - y_true_onehot) / batch_size

    # Gradient w.r.t. W and b
    dL_dW = x.T @ dL_dlogits
    dL_db = np.sum(dL_dlogits, axis=0)

    # ==================== PARAMETER UPDATE ====================
    W -= lr * dL_dW
    b -= lr * dL_db

    if epoch % 20 == 0:
        # Compute accuracy
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y_true)
        assert_loss(epoch, loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
