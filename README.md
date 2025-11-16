# PyTorch Tutorial

A tutorial project exploring PyTorch's internal architecture and automatic differentiation system.

## Documentation

This repository contains detailed analyses of PyTorch's autograd system:

- **[docs/backward.md](docs/backward.md)** - Comprehensive analysis of how PyTorch's `backward()` method works
  - Dynamic computational graph traversal
  - C++ engine architecture
  - Chain rule implementation
  - Gradient accumulation

- **[docs/forward.md](docs/forward.md)** - Comprehensive analysis of how PyTorch's `forward()` method works
  - Module `__call__` mechanism
  - Computational graph construction
  - Backward node creation
  - Hooks system

## Sample Code

- **[sample.py](sample.py)** - Simple neural network example demonstrating forward and backward passes
