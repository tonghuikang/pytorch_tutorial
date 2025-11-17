# NumPy & PyTorch Essential Syntax Reference

Quick reference for common operations in NumPy and PyTorch, presented side-by-side.

## Table of Contents

1. [Array/Tensor Creation](#1-arraytensor-creation)
2. [Element-wise Operations](#2-element-wise-operations)
3. [Matrix Multiplication: `@`](#3-matrix-multiplication-)
4. [Transpose: `.T`](#4-transpose-t)
5. [Reshaping](#5-reshaping)
6. [Indexing & Slicing](#6-indexing--slicing)
7. [Aggregation Functions](#7-aggregation-functions)
8. [Broadcasting](#8-broadcasting)
9. [Linear Algebra](#9-linear-algebra)
10. [Common Activations](#10-common-activations)

---

## 1. Array/Tensor Creation

### NumPy
```python
# From list
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])

# Special arrays
np.zeros((2, 3))           # 2×3 zeros
np.ones((3, 4))            # 3×4 ones
np.eye(3)                  # 3×3 identity
np.full((2, 2), 7)         # 2×2 filled with 7

# Ranges
np.arange(10)              # [0, 1, ..., 9]
np.arange(2, 10, 2)        # [2, 4, 6, 8]
np.linspace(0, 1, 5)       # [0, 0.25, 0.5, 0.75, 1]

# Random
np.random.randn(3, 4)      # 3×4 from N(0,1)
np.random.rand(3, 4)       # 3×4 uniform [0,1)
np.random.randint(0, 10, size=5)  # 5 random ints

# Data types
np.array([1, 2, 3], dtype=np.float32)
np.array([1, 2, 3], dtype=np.int64)
```

### PyTorch
```python
# From list
torch.tensor([1, 2, 3])
torch.tensor([[1, 2], [3, 4]])

# Special tensors
torch.zeros(2, 3)          # 2×3 zeros
torch.ones(3, 4)           # 3×4 ones
torch.eye(3)               # 3×3 identity
torch.full((2, 2), 7)      # 2×2 filled with 7

# Ranges
torch.arange(10)           # tensor([0, 1, ..., 9])
torch.arange(2, 10, 2)     # tensor([2, 4, 6, 8])
torch.linspace(0, 1, 5)    # tensor([0, 0.25, ..., 1])

# Random
torch.randn(3, 4)          # 3×4 from N(0,1)
torch.rand(3, 4)           # 3×4 uniform [0,1)
torch.randint(0, 10, size=(5,))  # 5 random ints

# Data types
torch.tensor([1, 2, 3], dtype=torch.float32)
torch.tensor([1, 2, 3], dtype=torch.int64)

# From NumPy
torch.from_numpy(np_array)
tensor.numpy()             # Tensor to NumPy
```

---

## 2. Element-wise Operations

Operations applied element-by-element (not matrix operations).

### NumPy
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Arithmetic
a + b              # [5, 7, 9]
a - b              # [-3, -3, -3]
a * b              # [4, 10, 18] - element-wise multiplication
a / b              # [0.25, 0.4, 0.5]
a ** 2             # [1, 4, 9] - power

# With scalars
a + 10             # [11, 12, 13]
a * 2              # [2, 4, 6]

# Math functions
np.sqrt(a)         # [1., 1.414, 1.732]
np.exp(a)          # [e^1, e^2, e^3]
np.log(a)          # [0, 0.693, 1.099]

# Comparisons
a > 2              # [False, False, True]
a == 2             # [False, True, False]
```

### PyTorch
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Arithmetic
a + b              # tensor([5, 7, 9])
a - b              # tensor([-3, -3, -3])
a * b              # tensor([4, 10, 18]) - element-wise
a / b              # tensor([0.25, 0.4, 0.5])
a ** 2             # tensor([1, 4, 9])

# With scalars
a + 10             # tensor([11, 12, 13])
a * 2              # tensor([2, 4, 6])

# Math functions
torch.sqrt(a.float())  # tensor([1., 1.414, 1.732])
torch.exp(a.float())   # tensor([e^1, e^2, e^3])
torch.log(a.float())   # tensor([0., 0.693, 1.099])

# Comparisons
a > 2              # tensor([False, False, True])
a == 2             # tensor([False, True, False])
```

**Key Distinction**: `*` is element-wise, `@` is matrix multiplication.

---

## 3. Matrix Multiplication: `@`

The `@` operator performs matrix/vector multiplication.

### NumPy
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])           # 2×3 matrix
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])            # 3×2 matrix
v = np.array([1, 2, 3])             # 1D vector

# Matrix @ Matrix
A @ B              # [[58, 64], [139, 154]]
                   # [[1*7+2*9+3*11, 1*8+2*10+3*12],
                   #  [4*7+5*9+6*11, 4*8+5*10+6*12]]

# Matrix @ Vector
A @ v              # [14, 32]
                   # [1*1+2*2+3*3, 4*1+5*2+6*3]

# Vector @ Vector (dot product)
v @ v              # 14
                   # 1*1 + 2*2 + 3*3

# Alternative: np.dot()
np.dot(A, B)       # Same as A @ B
```

### PyTorch
```python
import torch

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])           # 2×3 tensor
B = torch.tensor([[7, 8],
                  [9, 10],
                  [11, 12]])            # 3×2 tensor
v = torch.tensor([1, 2, 3])             # 1D tensor

# Tensor @ Tensor
A @ B              # tensor([[58, 64], [139, 154]])
                   # [[1*7+2*9+3*11, 1*8+2*10+3*12],
                   #  [4*7+5*9+6*11, 4*8+5*10+6*12]]

# Tensor @ Vector
A @ v              # tensor([14, 32])
                   # [1*1+2*2+3*3, 4*1+5*2+6*3]

# Vector @ Vector (dot product)
v @ v              # tensor(14)
                   # 1*1 + 2*2 + 3*3

# Alternatives
torch.matmul(A, B)  # Same as A @ B
torch.mm(A, B)      # 2D matrices only
torch.dot(v, v)     # 1D tensors only
```

**Key Point**: `@` is the universal matrix multiplication operator in both libraries.

---

## 4. Transpose: `.T`

Swaps rows and columns (reverses all axes).

### NumPy
```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

A.T                # [[1, 4], [2, 5], [3, 6]]  # Shape: (3, 2)

# WARNING: 1D arrays don't transpose!
v = np.array([1, 2, 3])
v.T                # [1, 2, 3] - unchanged!

# To make column vector:
v[:, np.newaxis]   # [[1], [2], [3]] - Shape: (3, 1)
v[np.newaxis, :]   # [[1, 2, 3]] - Shape: (1, 3)
```

### PyTorch
```python
A = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

A.T                # tensor([[1, 4], [2, 5], [3, 6]])  # Shape: (3, 2)

# Alternative: .t() for 2D tensors only
A.t()              # Same as A.T

# For 1D tensors
v = torch.tensor([1, 2, 3])
v.T                # tensor([1, 2, 3]) - unchanged!

# To make column/row vector:
v.unsqueeze(1)     # tensor([[1], [2], [3]]) - Shape: (3, 1)
v.unsqueeze(0)     # tensor([[1, 2, 3]]) - Shape: (1, 3)
v[:, None]         # Same as unsqueeze(1)
```

---

## 5. Reshaping

### NumPy
```python
a = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
a.reshape(3, 4)    # 3×4 array
a.reshape(2, -1)   # 2×6 (auto-calculate)
a.reshape(-1)      # Flatten to 1D

# View shape
a = np.array([[1, 2], [3, 4]])
a.shape            # (2, 2)
a.ndim             # 2
a.size             # 4

# Add/remove dimensions
a.squeeze()        # Remove dimensions of size 1
a[:, np.newaxis]   # Add dimension
np.expand_dims(a, axis=0)

# Flatten
a.flatten()        # Returns copy
a.ravel()          # Returns view if possible
```

### PyTorch
```python
a = torch.arange(12)  # tensor([0, 1, ..., 11])

# Reshape
a.reshape(3, 4)    # 3×4 tensor
a.reshape(2, -1)   # 2×6 (auto-calculate)
a.reshape(-1)      # Flatten to 1D
a.view(3, 4)       # Like reshape (must be contiguous)

# View shape
a = torch.tensor([[1, 2], [3, 4]])
a.shape            # torch.Size([2, 2])
a.ndim             # 2
a.numel()          # 4 (number of elements)

# Add/remove dimensions
a.squeeze()        # Remove dimensions of size 1
a.unsqueeze(0)     # Add dimension at position 0
a[:, None]         # Add dimension (same as unsqueeze(1))

# Flatten
a.flatten()        # Flatten to 1D
a.ravel()          # Like flatten
```

---

## 6. Indexing & Slicing

### NumPy
```python
a = np.array([10, 20, 30, 40, 50])

# Basic indexing
a[0]               # 10 - first element
a[-1]              # 50 - last element
a[1:4]             # [20, 30, 40] - slice
a[::2]             # [10, 30, 50] - every 2nd
a[::-1]            # [50, 40, 30, 20, 10] - reverse

# 2D indexing
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A[0, 0]            # 1
A[1, :]            # [4, 5, 6] - row 1
A[:, 1]            # [2, 5, 8] - column 1
A[:2, :2]          # [[1, 2], [4, 5]] - top-left 2×2

# Boolean indexing
a[a > 25]          # [30, 40, 50]
a[a % 20 == 0]     # [20, 40]

# Modify with indexing
a[a > 30] = 0      # Replace elements > 30 with 0
```

### PyTorch
```python
a = torch.tensor([10, 20, 30, 40, 50])

# Basic indexing
a[0]               # tensor(10)
a[-1]              # tensor(50)
a[1:4]             # tensor([20, 30, 40])
a[::2]             # tensor([10, 30, 50])
a.flip(0)          # Reverse (different syntax!)

# 2D indexing
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A[0, 0]            # tensor(1)
A[1, :]            # tensor([4, 5, 6])
A[:, 1]            # tensor([2, 5, 8])
A[:2, :2]          # tensor([[1, 2], [4, 5]])

# Boolean indexing
a[a > 25]          # tensor([30, 40, 50])
a[a % 20 == 0]     # tensor([20, 40])

# Modify with indexing
a[a > 30] = 0      # Replace elements > 30 with 0
```

---

## 7. Aggregation Functions

### NumPy
```python
a = np.array([[1, 2, 3], [4, 5, 6]])

# Sum
np.sum(a)          # 21 - all elements
np.sum(a, axis=0)  # [5, 7, 9] - sum columns
np.sum(a, axis=1)  # [6, 15] - sum rows

# Mean
np.mean(a)         # 3.5
np.mean(a, axis=0) # [2.5, 3.5, 4.5]

# Min/Max
np.min(a)          # 1
np.max(a)          # 6
np.argmin(a)       # 0 - index of min (flattened)
np.argmax(a)       # 5 - index of max (flattened)

# Standard deviation
np.std(a)          # 1.707...
np.var(a)          # 2.916... - variance

# Other
a.sum()            # Method form
a.mean()
a.max()
```

### PyTorch
```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Sum
torch.sum(a)       # tensor(21)
torch.sum(a, dim=0)  # tensor([5, 7, 9]) - sum columns
torch.sum(a, dim=1)  # tensor([6, 15]) - sum rows

# Mean (requires float)
torch.mean(a.float())         # tensor(3.5)
torch.mean(a.float(), dim=0)  # tensor([2.5, 3.5, 4.5])

# Min/Max
torch.min(a)       # tensor(1)
torch.max(a)       # tensor(6)
torch.argmin(a)    # tensor(0)
torch.argmax(a)    # tensor(5)

# Standard deviation
torch.std(a.float())  # tensor(1.707...)
torch.var(a.float())  # tensor(2.916...)

# Other
a.sum()            # Method form
a.float().mean()
a.max()
```

**Note**: PyTorch uses `dim` instead of `axis`.

---

## 8. Broadcasting

Broadcasting allows operations on arrays/tensors of different shapes.

### NumPy
```python
# Scalar broadcasting
a = np.array([1, 2, 3])
a + 10             # [11, 12, 13]

# Row vector + matrix
A = np.ones((3, 4))
v = np.array([1, 2, 3, 4])
A + v              # Adds v to each row

# Column vector + matrix
v = np.array([[1], [2], [3]])  # (3, 1)
A + v              # Adds v to each column

# Broadcasting example
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])     # (3,)
a + b              # [[11, 21, 31],
                   #  [12, 22, 32],
                   #  [13, 23, 33]]
```

### PyTorch
```python
# Scalar broadcasting
a = torch.tensor([1, 2, 3])
a + 10             # tensor([11, 12, 13])

# Row vector + matrix
A = torch.ones(3, 4)
v = torch.tensor([1, 2, 3, 4])
A + v              # Adds v to each row

# Column vector + matrix
v = torch.tensor([[1], [2], [3]])  # (3, 1)
A + v              # Adds v to each column

# Broadcasting example
a = torch.tensor([[1], [2], [3]])  # (3, 1)
b = torch.tensor([10, 20, 30])     # (3,)
a + b              # tensor([[11, 21, 31],
                   #         [12, 22, 32],
                   #         [13, 23, 33]])
```

**Broadcasting Rule**: Dimensions are compatible if they're equal or one is 1.

---

## 9. Linear Algebra

### NumPy
```python
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Matrix operations
A @ A              # Matrix multiplication
A.T                # Transpose
np.trace(A)        # 5 - trace (sum of diagonal)

# Determinant and inverse
np.linalg.det(A)   # -2.0
np.linalg.inv(A)   # Inverse matrix

# Eigenvalues/vectors
eigenvals, eigenvecs = np.linalg.eig(A)

# Vector norm
np.linalg.norm(v)  # 2.236... (L2 norm)
np.linalg.norm(v, ord=1)  # L1 norm

# Solve Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)  # [1, 2]
```

### PyTorch
```python
A = torch.tensor([[1., 2.], [3., 4.]])
v = torch.tensor([1., 2.])

# Matrix operations
A @ A              # Matrix multiplication
A.T                # Transpose
torch.trace(A)     # tensor(5.) - trace

# Determinant and inverse
torch.linalg.det(A)   # tensor(-2.)
torch.linalg.inv(A)   # Inverse matrix

# Eigenvalues/vectors
eigenvals, eigenvecs = torch.linalg.eig(A)

# Vector norm
torch.linalg.norm(v)  # tensor(2.236...)
torch.linalg.norm(v, ord=1)  # L1 norm

# Solve Ax = b
b = torch.tensor([5., 11.])
x = torch.linalg.solve(A, b)  # tensor([1., 2.])
```

---

## 10. Common Activations

### NumPy
```python
x = np.array([-2, -1, 0, 1, 2])

# ReLU
np.maximum(0, x)   # [0, 0, 0, 1, 2]

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
np.tanh(x)         # [-0.96, -0.76, 0, 0.76, 0.96]

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
```

### PyTorch
```python
import torch.nn.functional as F

x = torch.tensor([-2., -1., 0., 1., 2.])

# ReLU
F.relu(x)          # tensor([0., 0., 0., 1., 2.])
torch.relu(x)      # Same

# Sigmoid
torch.sigmoid(x)   # tensor([0.12, 0.27, 0.5, 0.73, 0.88])

# Tanh
torch.tanh(x)      # tensor([-0.96, -0.76, 0., 0.76, 0.96])

# Softmax
F.softmax(x, dim=0)  # tensor([0.01, 0.03, 0.09, 0.24, 0.64])
```

---

## Quick Comparison Table

| Operation | NumPy | PyTorch |
|-----------|-------|---------|
| **Matrix multiply** | `A @ B` | `A @ B` |
| **Transpose** | `A.T` | `A.T` or `A.t()` |
| **Element-wise** | `A * B` | `A * B` |
| **Create zeros** | `np.zeros((2, 3))` | `torch.zeros(2, 3)` |
| **Reshape** | `a.reshape(2, -1)` | `a.reshape(2, -1)` or `a.view(2, -1)` |
| **Add dimension** | `a[:, np.newaxis]` | `a.unsqueeze(1)` or `a[:, None]` |
| **Sum over axis** | `np.sum(a, axis=0)` | `torch.sum(a, dim=0)` |
| **Random normal** | `np.random.randn(3, 4)` | `torch.randn(3, 4)` |
| **Inverse** | `np.linalg.inv(A)` | `torch.linalg.inv(A)` |
| **ReLU** | `np.maximum(0, x)` | `F.relu(x)` |

---

## Key Differences & Common Footguns

### 1. **`axis` vs `dim`**
```python
# NumPy uses 'axis'
np.sum(a, axis=0)
np.mean(a, axis=1)

# PyTorch uses 'dim'
torch.sum(a, dim=0)
torch.mean(a, dim=1)
```

### 2. **Type Requirements**
```python
# NumPy is flexible with integer types
a = np.array([1, 2, 3])
np.mean(a)  # Works: 2.0

# PyTorch requires float for some operations
a = torch.tensor([1, 2, 3])
torch.mean(a)  # ERROR! Needs float
torch.mean(a.float())  # Works: tensor(2.)
```

### 3. **In-place Operations**
```python
# NumPy: most operations return new arrays
a = np.array([1, 2, 3])
b = a + 1  # New array

# PyTorch: underscore suffix = in-place
a = torch.tensor([1, 2, 3])
a.add_(1)  # Modifies a in-place!
a.relu_()  # In-place ReLU
```

### 4. **Random Number Generation**
```python
# NumPy: global state (legacy)
np.random.seed(42)
np.random.randn(3, 4)

# PyTorch: use generators (preferred)
generator = torch.Generator().manual_seed(42)
torch.randn(3, 4, generator=generator)

# Or global (not recommended)
torch.manual_seed(42)
torch.randn(3, 4)
```

### 5. **Copy vs View**
```python
# NumPy: slicing creates views
a = np.array([1, 2, 3, 4])
b = a[1:3]  # View
b[0] = 99   # Modifies a too!

# PyTorch: same behavior!
a = torch.tensor([1, 2, 3, 4])
b = a[1:3]  # View
b[0] = 99   # Also modifies a!

# Both: use .copy() or .clone()
b = a[1:3].copy()   # NumPy
b = a[1:3].clone()  # PyTorch
```

### 6. **Device Management (GPU)**
```python
# NumPy: CPU only
a = np.array([1, 2, 3])

# PyTorch: explicit device management
a = torch.tensor([1, 2, 3])  # CPU by default
a_gpu = a.to('cuda')  # Move to GPU
a_gpu = a.cuda()      # Alternative

# FOOTGUN: Operations require same device!
a_cpu = torch.tensor([1, 2, 3])
a_gpu = torch.tensor([4, 5, 6]).cuda()
a_cpu + a_gpu  # ERROR! Different devices
```

### 7. **Gradient Tracking**
```python
# PyTorch: automatic differentiation
a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a * 2
b.sum().backward()  # Computes gradients
print(a.grad)  # tensor([2., 2., 2.])

# FOOTGUN: In-place ops break gradient tracking!
a = torch.tensor([1., 2., 3.], requires_grad=True)
a += 1  # ERROR or warning!
```

### 8. **Boolean Indexing Returns**
```python
# NumPy: returns array
a = np.array([1, 2, 3, 4])
result = a[a > 2]  # array([3, 4])

# PyTorch: returns tensor (same behavior)
a = torch.tensor([1, 2, 3, 4])
result = a[a > 2]  # tensor([3, 4])
```

### 9. **Reverse/Flip**
```python
# NumPy: negative stride
a = np.array([1, 2, 3, 4])
a[::-1]  # [4, 3, 2, 1]

# PyTorch: use flip() method
a = torch.tensor([1, 2, 3, 4])
a.flip(0)  # tensor([4, 3, 2, 1])
# a[::-1] doesn't work in PyTorch!
```

### 10. **Matrix Multiplication Variants**
```python
# NumPy
np.dot(a, b)      # General dot product
a @ b             # Matrix multiplication (preferred)

# PyTorch
torch.matmul(a, b)  # General, handles broadcasting
torch.mm(a, b)      # Only 2D matrices
torch.bmm(a, b)     # Batch matrix multiplication
a @ b               # Same as matmul (preferred)

# FOOTGUN: torch.mm doesn't broadcast!
A = torch.randn(2, 3)
B = torch.randn(3, 4, 5)
torch.matmul(A, B)  # Works: (2, 4, 5)
torch.mm(A, B)      # ERROR! mm requires 2D
```

### 11. **Reshape vs View**
```python
# NumPy: reshape always works
a = np.arange(6)
a.reshape(2, 3)  # Always succeeds

# PyTorch: view requires contiguous memory
a = torch.arange(6)
a.view(2, 3)      # Works (contiguous)
a.t().view(6)     # ERROR! Not contiguous
a.t().reshape(6)  # Works (handles non-contiguous)
```

### 12. **Size vs Shape**
```python
# NumPy: .shape and .size
a = np.array([[1, 2], [3, 4]])
a.shape  # (2, 2)
a.size   # 4 (total elements)

# PyTorch: .shape, .size(), .numel()
a = torch.tensor([[1, 2], [3, 4]])
a.shape    # torch.Size([2, 2])
a.size()   # torch.Size([2, 2]) - it's a method!
a.numel()  # 4 (total elements)
```
