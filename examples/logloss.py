"""
Logistic regression with cross-entropy loss for binary classification.

logits = x @ W + b
probs = softmax(logits)
L = cross_entropy(probs, y_true)

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
from assertion import assert_loss

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
    0: 1.074304,
    20: 0.985010,
    40: 0.907743,
    60: 0.841064,
    80: 0.783557,
}


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
        assert_loss(epoch, loss.item(), epoch_to_expected_loss)
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
    # Cross-entropy loss with softmax has an elegant gradient:
    # L = -mean(log(probs[correct_class]))
    #
    # Softmax: probs[i,j] = exp(logits[i,j]) / sum_k(exp(logits[i,k]))
    #
    # The derivative d/dlogits[i,j] of the cross-entropy+softmax is:
    # dL/dlogits[i,j] = probs[i,j] - y_true_onehot[i,j]
    #
    # Derivation:
    # 1. dL/dlogits[i,j] = d/dlogits[i,j][-log(probs[i, y_true[i]])]
    # 2. Apply chain rule: = -(1/probs[i, y_true[i]]) * dprobs[i, y_true[i]]/dlogits[i,j]
    # 3. For softmax derivative:
    #    - If j == y_true[i]: dprobs[i,j]/dlogits[i,j] = probs[i,j] * (1 - probs[i,j])
    #    - If j != y_true[i]: dprobs[i,j]/dlogits[i,j] = -probs[i,j] * probs[i,y_true[i]]
    # 4. After simplification (canceling terms), both cases give: probs[i,j] - y_true_onehot[i,j]
    # 5. Dividing by batch_size because loss uses mean

    # Create one-hot encoding of true labels (with explicit float32 dtype)
    y_true_onehot = np.zeros((batch_size, num_classes), dtype=np.float32)
    y_true_onehot[np.arange(batch_size), y_true] = 1

    # Gradient of loss w.r.t. logits (the elegant result!)
    dL_dlogits = (probs - y_true_onehot) / batch_size

    # Backprop to weights and bias: logits = x @ W + b
    # dL/dW = x.T @ dL/dlogits
    # x.T has shape [input_dim, batch], dL/dlogits has shape [batch, num_classes]
    # Result: [input_dim, num_classes]
    dL_dW = x.T @ dL_dlogits
    # dL/db: sum gradient over batch dimension
    dL_db = np.sum(dL_dlogits, axis=0)

    # ==================== PARAMETER UPDATE ====================
    W -= lr * dL_dW
    b -= lr * dL_db

    if epoch % 20 == 0:
        # Compute accuracy
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y_true)
        assert_loss(epoch, loss, epoch_to_expected_loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
