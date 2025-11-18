"""
Scaled dot-product attention mechanism.

(Untrainable)
      X                                                 y_true
      │                                                   |
      ▼                                                   ▼
┌───────────────┐            ┌───────────┐         ┌───────────────┐
│ X @ W_{q,k,v} │──→ Q,K,V──→│ attention │──→ y ──→│ (y, y_true)^2 │──→ L
└───────────────┘            └───────────┘         └───────────────┘
      ▲
      │
   Wq,Wk,Wv
(Trainable)

Implementations:
1. Manual PyTorch (with explicit scores and weights)
2. PyTorch built-in (using F.scaled_dot_product_attention)
3. NumPy (manual gradients)

Dimensions:
  X:  (batch_size, seq_length, d_model)  e.g., (2, 4, 8)
  Q:  (batch_size, seq_length, d_k)      e.g., (2, 4, 4)
  K:  (batch_size, seq_length, d_k)      e.g., (2, 4, 4)
  V:  (batch_size, seq_length, d_k)      e.g., (2, 4, 4)
  y:  (batch_size, seq_length, d_k)      e.g., (2, 4, 4)
"""

import torch
import numpy as np
import math

# Set seed for reproducibility
np.random.seed(42)

# Set dimensions
batch_size = 2
seq_length = 4
d_model = 8
d_k = 4
lr = 0.001

# ============================================================================
# Generate Initial Variables (used by both PyTorch and NumPy)
# ============================================================================

# Initialize parameters
W_q_initial = np.random.randn(d_model, d_k).astype(np.float32)
W_k_initial = np.random.randn(d_model, d_k).astype(np.float32)
W_v_initial = np.random.randn(d_model, d_k).astype(np.float32)

# Input sequence
X_data = np.random.randn(batch_size, seq_length, d_model).astype(np.float32)
Y_true_data = np.random.randn(batch_size, seq_length, d_k).astype(np.float32)

# ============================================================================
# Expected Loss Values (for testing)
# ============================================================================

epoch_to_expected_loss = {
    0: 4.6504,
    10: 4.5517,
    20: 4.4572,
    30: 4.3664,
    40: 4.2792,
}


def assert_loss(epoch: int, loss: float) -> None:
    if epoch in epoch_to_expected_loss:
        assert abs(loss - epoch_to_expected_loss[epoch]) < 0.001, (
            f"Epoch {epoch}: expected {epoch_to_expected_loss[epoch]}, got {loss}"
        )

# ============================================================================
# PyTorch Built-in scaled_dot_product_attention
# ============================================================================

W_q = torch.tensor(W_q_initial.copy(), dtype=torch.float32, requires_grad=True)
W_k = torch.tensor(W_k_initial.copy(), dtype=torch.float32, requires_grad=True)
W_v = torch.tensor(W_v_initial.copy(), dtype=torch.float32, requires_grad=True)

X = torch.tensor(X_data, dtype=torch.float32)
Y_true = torch.tensor(Y_true_data, dtype=torch.float32)

optimizer = torch.optim.SGD([W_q, W_k, W_v], lr=lr)

print("=" * 60)
print("PyTorch Scaled Dot-Product Attention (Built-in)")
print("=" * 60)

for epoch in range(50):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    context = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    loss = torch.mean((context - Y_true) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        assert_loss(epoch, loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# PyTorch Manual Implementation
# ============================================================================

W_q = torch.tensor(W_q_initial.copy(), dtype=torch.float32, requires_grad=True)
W_k = torch.tensor(W_k_initial.copy(), dtype=torch.float32, requires_grad=True)
W_v = torch.tensor(W_v_initial.copy(), dtype=torch.float32, requires_grad=True)

X = torch.tensor(X_data, dtype=torch.float32)
Y_true = torch.tensor(Y_true_data, dtype=torch.float32)

optimizer = torch.optim.SGD([W_q, W_k, W_v], lr=lr)

print("=" * 60)
print("PyTorch Scaled Dot-Product Attention (Manual)")
print("=" * 60)

for epoch in range(50):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ V

    loss = torch.mean((context - Y_true) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        assert_loss(epoch, loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# NumPy Implementation
# ============================================================================

W_q = W_q_initial.copy()
W_k = W_k_initial.copy()
W_v = W_v_initial.copy()

X = X_data.copy()
Y_true = Y_true_data.copy()

print("=" * 60)
print("NumPy Scaled Dot-Product Attention (Manual Gradients)")
print("=" * 60)


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


for epoch in range(50):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(d_k)
    weights = softmax(scores)
    context = weights @ V

    diff = context - Y_true
    loss = np.mean(diff**2)

    dL_dcontext = (2.0 / (batch_size * seq_length * d_k)) * diff

    dL_dV = weights.transpose(0, 2, 1) @ dL_dcontext
    dL_dweights = dL_dcontext @ V.transpose(0, 2, 1)

    sum_term = np.sum(weights * dL_dweights, axis=-1, keepdims=True)
    dL_dscores = weights * (dL_dweights - sum_term)

    dL_dscores = dL_dscores / math.sqrt(d_k)

    dL_dQ = dL_dscores @ K
    dL_dK = dL_dscores.transpose(0, 2, 1) @ Q

    dL_dW_q = np.zeros_like(W_q)
    dL_dW_k = np.zeros_like(W_k)
    dL_dW_v = np.zeros_like(W_v)

    for i in range(batch_size):
        dL_dW_q += X[i].T @ dL_dQ[i]
        dL_dW_k += X[i].T @ dL_dK[i]
        dL_dW_v += X[i].T @ dL_dV[i]

    W_q -= lr * dL_dW_q
    W_k -= lr * dL_dW_k
    W_v -= lr * dL_dW_v

    if epoch % 10 == 0:
        assert_loss(epoch, loss)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
