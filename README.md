# PyTorch Tutorial

A tutorial project exploring PyTorch's internal architecture and automatic differentiation system.

## Documentation

This repository contains detailed analyses of PyTorch's internal architecture:

- **[docs/syntax.md](docs/syntax.md)** - Essential syntax reference for NumPy and PyTorch
  - Matrix multiplication (`@`), transpose (`.T`), element-wise operations
  - Array/tensor creation, reshaping, indexing, and slicing
  - Broadcasting rules and aggregation functions
  - Linear algebra operations and common activations
  - Side-by-side comparison with key differences highlighted

- **[docs/tensor.md](docs/tensor.md)** - Comprehensive analysis of how PyTorch stores tensor data
  - Storage vs tensor metadata (shape, strides, offset)
  - Storage sharing and view mechanics
  - Contiguous vs non-contiguous tensors
  - C++ implementation (StorageImpl, TensorImpl, DataPtr)
  - CPU vs CUDA memory allocation
  - Memory pooling and caching

- **[docs/forward.md](docs/forward.md)** - Comprehensive analysis of how PyTorch's `forward()` method works
  - Module `__call__` mechanism
  - Computational graph construction
  - Backward node creation
  - Hooks system

- **[docs/backward.md](docs/backward.md)** - Comprehensive analysis of how PyTorch's `backward()` method works
  - Dynamic computational graph traversal
  - C++ engine architecture
  - Chain rule implementation
  - Gradient accumulation

- **[docs/numpy.md](docs/numpy.md)** - Comparison of backpropagation: PyTorch vs NumPy
  - Manual gradient derivation and implementation
  - PyTorch automatic differentiation vs NumPy manual computation
  - Scalar and matrix implementations
  - Common pitfalls in manual backpropagation
  - Performance and feature comparisons

- **[docs/optimizer.md](docs/optimizer.md)** - Comprehensive analysis of how PyTorch's `optimizer.step()` works
  - Optimizer base class architecture
  - Adam/AdamW algorithm implementation
  - State management and parameter groups
  - Foreach and fused execution modes
  - Learning rate schedulers

- **[docs/compile.md](docs/compile.md)** - Comprehensive analysis of how PyTorch's `torch.compile()` works
  - TorchDynamo graph capture system
  - Bytecode analysis and symbolic execution
  - TorchInductor backend and optimization
  - Guards system and recompilation
  - Dynamic shapes handling

## Sample Code

- **[sample.py](sample.py)** - Simple neural network example demonstrating forward and backward passes
