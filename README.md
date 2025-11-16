# PyTorch Tutorial

A tutorial project exploring PyTorch's internal architecture and automatic differentiation system.

## Documentation

This repository contains detailed analyses of PyTorch's internal architecture:

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

- **[docs/compile.md](docs/compile.md)** - Comprehensive analysis of how PyTorch's `torch.compile()` works
  - TorchDynamo graph capture system
  - Bytecode analysis and symbolic execution
  - TorchInductor backend and optimization
  - Guards system and recompilation
  - Dynamic shapes handling

- **[docs/optimizer.md](docs/optimizer.md)** - Comprehensive analysis of how PyTorch's `optimizer.step()` works
  - Optimizer base class architecture
  - Adam/AdamW algorithm implementation
  - State management and parameter groups
  - Foreach and fused execution modes
  - Learning rate schedulers

## Sample Code

- **[sample.py](sample.py)** - Simple neural network example demonstrating forward and backward passes
