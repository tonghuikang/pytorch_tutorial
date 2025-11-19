"""
Residual networks (ResNets) with skip connections.

a1 = x @ W1 + b1
z1 = ReLU(a1)
h = z1 + x
a2 = h @ W2 + b2
z2 = ReLU(a2)
y = z2 + h
L = mean((y - y_true)^2)

(Untrainable)
       x                                                                                                                          y_true
       ├────────────────────────────────────────┐        ┌─────────────────────────────────────────────────────────────┐            │
       ▼                                        ▼        │                                                             ▼            ▼
┌─────────────┐          ┌──────┐          ┌────────┐         ┌─────────────┐          ┌──────┐          ┌────────┐         ┌───────────────┐   ┌──────┐
│ x @ W1 + b1 │──→ a1 ──→│ ReLU │──→ z1 ──→│ z1 + x │──→ h ──→│ h @ W2 + b2 │──→ a2 ──→│ ReLU │──→ z2 ──→│ z2 + h │──→ y ──→│ (y, y_true)^2 │──→│ mean │──→ L (loss)
└─────────────┘          └──────┘          └────────┘         └─────────────┘          └──────┘          └────────┘         └───────────────┘   └──────┘
       ▲                                                             ▲
       │                                                             │
     W1,b1                                                         W2,b2
(Trainable)

Dimensions:
  x:      (batch, input_dim)  e.g., (8, 8)
  y_true: (batch, output_dim) e.g., (8, 8)

  W1:     (input_dim, hidden_dim)  e.g., (8, 8)
  b1:     (hidden_dim,)            e.g., (8,)
  W2:     (hidden_dim, hidden_dim) e.g., (8, 8)
  b2:     (hidden_dim,)            e.g., (8,)

  h:      (batch, hidden_dim) e.g., (8, 8)
  y:      (batch, output_dim) e.g., (8, 8)
  loss:   scalar

Key innovation: Skip connections enable deeper networks by:
- Providing direct gradient flow paths
- Reducing vanishing gradient problem
- Allowing residual learning
"""

import torch
import numpy as np
from assertion import assert_loss

# Set seed for reproducibility
np.random.seed(42)

# Set dimensions
batch_size = 8
input_dim = 8  # Must equal hidden_dim for h = ReLU(x @ W1 + b1) + x to work
hidden_dim = 8  # Must equal input_dim and output_dim for residual connections
output_dim = 8
lr = 0.01

# ============================================================================
# Generate Initial Variables (used by both PyTorch and NumPy)
# ============================================================================

# Initialize parameters
W1_initial = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
b1_initial = np.random.randn(hidden_dim).astype(np.float32) * 0.1

W2_initial = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
b2_initial = np.random.randn(hidden_dim).astype(np.float32) * 0.1

# Input and target
x_data = np.random.randn(batch_size, input_dim).astype(np.float32)
y_true_data = np.random.randn(batch_size, output_dim).astype(np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 2.133458,
    20: 1.932017,
    40: 1.809436,
    60: 1.739568,
    80: 1.680289,
}


# ============================================================================
# PyTorch Implementation
# ============================================================================

W1 = torch.tensor(W1_initial.copy(), dtype=torch.float32, requires_grad=True)
b1 = torch.tensor(b1_initial.copy(), dtype=torch.float32, requires_grad=True)

W2 = torch.tensor(W2_initial.copy(), dtype=torch.float32, requires_grad=True)
b2 = torch.tensor(b2_initial.copy(), dtype=torch.float32, requires_grad=True)

x = torch.tensor(x_data, dtype=torch.float32)
y_true = torch.tensor(y_true_data, dtype=torch.float32)

params = [W1, b1, W2, b2]
optimizer = torch.optim.SGD(params, lr=lr)

print("=" * 60)
print("PyTorch Residual Network (with Skip Connections)")
print("=" * 60)

for epoch in range(100):
    # a1 = x @ W1 + b1
    a1 = x @ W1 + b1  # [batch_size, hidden_dim]
    # z1 = ReLU(a1)
    z1 = torch.relu(a1)  # [batch_size, hidden_dim]
    # h = z1 + x
    h = z1 + x  # [batch_size, hidden_dim] (residual connection)

    # a2 = h @ W2 + b2
    a2 = h @ W2 + b2  # [batch_size, hidden_dim]
    # z2 = ReLU(a2)
    z2 = torch.relu(a2)  # [batch_size, hidden_dim]
    # y = z2 + h
    y = z2 + h  # [batch_size, hidden_dim] (residual connection)

    loss = torch.mean((y - y_true) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        assert_loss(epoch, loss.item(), epoch_to_expected_loss)
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# NumPy Implementation
# ============================================================================

W1 = W1_initial.copy()
b1 = b1_initial.copy()

W2 = W2_initial.copy()
b2 = b2_initial.copy()

x = x_data.copy()
y_true = y_true_data.copy()

print("=" * 60)
print("NumPy Residual Network (Manual Gradients)")
print("=" * 60)

for epoch in range(100):
    # ==================== FORWARD PASS ====================
    # a1 = x @ W1 + b1
    a1 = x @ W1 + b1  # [batch_size, hidden_dim]
    # z1 = ReLU(a1)
    z1 = np.maximum(0, a1)  # [batch_size, hidden_dim]
    # h = z1 + x
    h = z1 + x  # [batch_size, hidden_dim] (residual connection)

    # a2 = h @ W2 + b2
    a2 = h @ W2 + b2  # [batch_size, hidden_dim]
    # z2 = ReLU(a2)
    z2 = np.maximum(0, a2)  # [batch_size, hidden_dim]
    # y = z2 + h
    y = z2 + h  # [batch_size, hidden_dim] (residual connection)

    diff = y - y_true
    loss = np.mean(diff**2)

    # ==================== BACKWARD PASS ====================
    # Loss: L = mean((y - y_true)^2)
    # dL/dy = 2(y - y_true) / (batch_size * hidden_dim)
    dL_dy = (2.0 / (batch_size * hidden_dim)) * diff

    # Backward through second addition: y = z2 + h
    # Both paths z2 and h receive gradient dL/dy
    dL_dz2 = dL_dy  # gradient for z2 (already post-ReLU)
    dL_dh_direct_2 = dL_dy  # gradient for direct h from residual connection

    # Backward through second ReLU: z2 = ReLU(a2)
    # ReLU gradient: d/da2[ReLU(a2)] = 1 if a2 > 0 else 0
    dL_da2 = dL_dz2 * (a2 > 0)  # apply ReLU mask

    # Backward through second linear layer: a2 = h @ W2 + b2
    # dL/dW2 = h.T @ dL/da2: [hidden_dim, batch] @ [batch, hidden_dim] = [hidden_dim, hidden_dim]
    dL_dW2 = h.T @ dL_da2
    # dL/db2: sum gradient over batch dimension
    dL_db2 = np.sum(dL_da2, axis=0)
    # Gradient flowing back to h from W2: dL/dh_from_W2 = dL/da2 @ W2.T
    dL_dh_from_W2 = dL_da2 @ W2.T

    # Combine gradients at h from both paths:
    # h receives gradient from: (1) W2 backward pass, (2) direct residual from y
    dL_dh = dL_dh_from_W2 + dL_dh_direct_2

    # Backward through first addition: h = z1 + x
    # Both paths z1 and x receive gradient dL_dh
    dL_dz1 = dL_dh  # gradient for z1 (already post-ReLU)
    dL_dx_direct_1 = dL_dh  # gradient for direct x from residual connection

    # Backward through first ReLU: z1 = ReLU(a1)
    # ReLU gradient: d/da1[ReLU(a1)] = 1 if a1 > 0 else 0
    dL_da1 = dL_dz1 * (a1 > 0)  # apply ReLU mask

    # Backward through first linear layer: a1 = x @ W1 + b1
    # dL/dW1 = x.T @ dL/da1: [input_dim, batch] @ [batch, hidden_dim] = [input_dim, hidden_dim]
    dL_dW1 = x.T @ dL_da1
    # dL/db1: sum gradient over batch dimension
    dL_db1 = np.sum(dL_da1, axis=0)
    # Gradient flowing back to x from W1: dL/dx_from_W1 = dL/da1 @ W1.T
    dL_dx_from_W1 = dL_da1 @ W1.T

    # Combine gradients at x from both paths:
    # x receives gradient from: (1) W1 backward pass, (2) direct residual from h
    dL_dx = dL_dx_from_W1 + dL_dx_direct_1

    # ==================== PARAMETER UPDATE ====================
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

    if epoch % 20 == 0:
        assert_loss(epoch, loss, epoch_to_expected_loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
