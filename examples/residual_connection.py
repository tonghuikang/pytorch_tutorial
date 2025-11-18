"""
Residual networks (ResNets) with skip connections.

                                                               y_true
    Main Path:                                                  |
    x ──→ [W1,b1] ──→ ReLU ──→ h1 ──→ [W2,b2] ──→ ReLU ──→ h2 ──→ [W3,b3] ──→ y_main
    │                                                            │         ▲
    │                                                            ▼         │
    │                                        ┌───────────────────────┐    │
    │                                        │      Add (+)          │    │
    │                                        └───────────────────────┘    │
    │                                                    ▲                │
    │  Skip Path:                                       │                │
    └──→ [W_skip,b_skip] ──────────────────────────────→┘         ┌──────────────┐
                                                                   │ MSE Loss: L  │
                                                            y ──→  └──────────────┘

Dimensions:
  x:      (batch, input_dim)     e.g., (8, 16)
  h1:     (batch, hidden_dim)    e.g., (8, 32)
  h2:     (batch, hidden_dim)    e.g., (8, 32)
  y_main: (batch, output_dim)    e.g., (8, 8)
  y_skip: (batch, output_dim)    e.g., (8, 8)
  y:      (batch, output_dim)    e.g., (8, 8)
  y_true: (batch, output_dim)    e.g., (8, 8)

Key innovation: Skip connections enable deeper networks by:
- Providing direct gradient flow paths
- Reducing vanishing gradient problem
- Allowing residual learning
"""

import torch
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Set dimensions
batch_size = 8
input_dim = 16
hidden_dim = 32
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

W3_initial = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
b3_initial = np.random.randn(output_dim).astype(np.float32) * 0.1

W_skip_initial = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
b_skip_initial = np.random.randn(output_dim).astype(np.float32) * 0.1

# Input and target
x_data = np.random.randn(batch_size, input_dim).astype(np.float32)
y_true_data = np.random.randn(batch_size, output_dim).astype(np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 1.2536,
    20: 0.9628,
    40: 0.7568,
    60: 0.6087,
    80: 0.4997,
}


def assert_loss(epoch: int, loss: float) -> None:
    if epoch in epoch_to_expected_loss:
        assert abs(loss - epoch_to_expected_loss[epoch]) < 0.001, (
            f"Epoch {epoch}: expected {epoch_to_expected_loss[epoch]}, got {loss}"
        )


# ============================================================================
# PyTorch Implementation
# ============================================================================

W1 = torch.tensor(W1_initial.copy(), dtype=torch.float32, requires_grad=True)
b1 = torch.tensor(b1_initial.copy(), dtype=torch.float32, requires_grad=True)

W2 = torch.tensor(W2_initial.copy(), dtype=torch.float32, requires_grad=True)
b2 = torch.tensor(b2_initial.copy(), dtype=torch.float32, requires_grad=True)

W3 = torch.tensor(W3_initial.copy(), dtype=torch.float32, requires_grad=True)
b3 = torch.tensor(b3_initial.copy(), dtype=torch.float32, requires_grad=True)

W_skip = torch.tensor(W_skip_initial.copy(), dtype=torch.float32, requires_grad=True)
b_skip = torch.tensor(b_skip_initial.copy(), dtype=torch.float32, requires_grad=True)

x = torch.tensor(x_data, dtype=torch.float32)
y_true = torch.tensor(y_true_data, dtype=torch.float32)

params = [W1, b1, W2, b2, W3, b3, W_skip, b_skip]
optimizer = torch.optim.SGD(params, lr=lr)

print("=" * 60)
print("PyTorch Residual Network (with Skip Connections)")
print("=" * 60)

for epoch in range(100):
    # Main path
    h1 = torch.relu(x @ W1 + b1)
    h2 = torch.relu(h1 @ W2 + b2)
    y_main = h2 @ W3 + b3

    # Skip connection
    y_skip = x @ W_skip + b_skip

    # Add residual
    y = y_main + y_skip

    loss = torch.mean((y - y_true) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        assert_loss(epoch, loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# NumPy Implementation
# ============================================================================

W1 = W1_initial.copy()
b1 = b1_initial.copy()

W2 = W2_initial.copy()
b2 = b2_initial.copy()

W3 = W3_initial.copy()
b3 = b3_initial.copy()

W_skip = W_skip_initial.copy()
b_skip = b_skip_initial.copy()

x = x_data.copy()
y_true = y_true_data.copy()

print("=" * 60)
print("NumPy Residual Network (Manual Gradients)")
print("=" * 60)

for epoch in range(100):
    # ==================== FORWARD PASS ====================
    # Main path
    h1 = np.maximum(0, x @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    y_main = h2 @ W3 + b3

    # Skip connection
    y_skip = x @ W_skip + b_skip

    # Add residual
    y = y_main + y_skip

    diff = y - y_true
    loss = np.mean(diff**2)

    # ==================== BACKWARD PASS ====================
    dL_dy = (2.0 / (batch_size * output_dim)) * diff

    # Split gradient between main path and skip path
    dL_dy_main = dL_dy
    dL_dy_skip = dL_dy

    # Backward through skip connection
    dL_dW_skip = x.T @ dL_dy_skip
    dL_db_skip = np.sum(dL_dy_skip, axis=0)
    dL_dx_skip = dL_dy_skip @ W_skip.T

    # Backward through main path (3 layers with ReLU)
    dL_dW3 = h2.T @ dL_dy_main
    dL_db3 = np.sum(dL_dy_main, axis=0)
    dL_dh2 = dL_dy_main @ W3.T

    # Through ReLU
    dL_dh2 = dL_dh2 * (h2 > 0)

    dL_dW2 = h1.T @ dL_dh2
    dL_db2 = np.sum(dL_dh2, axis=0)
    dL_dh1 = dL_dh2 @ W2.T

    # Through ReLU
    dL_dh1 = dL_dh1 * (h1 > 0)

    dL_dW1 = x.T @ dL_dh1
    dL_db1 = np.sum(dL_dh1, axis=0)
    dL_dx_main = dL_dh1 @ W1.T

    # Combine gradients from skip and main paths
    dL_dx = dL_dx_main + dL_dx_skip

    # ==================== PARAMETER UPDATE ====================
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W3 -= lr * dL_dW3
    b3 -= lr * dL_db3
    W_skip -= lr * dL_dW_skip
    b_skip -= lr * dL_db_skip

    if epoch % 20 == 0:
        assert_loss(epoch, loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
