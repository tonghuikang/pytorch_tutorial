# PyTorch's backward() Architecture

This document explains how PyTorch's `backward()` method works internally, based on investigation of the PyTorch source code in `.venv`.

## Overview

PyTorch's automatic differentiation system uses a **dynamic computational graph** where operations are recorded as they happen. When `backward()` is called, the system traverses this graph in reverse topological order, applying the chain rule to compute gradients.

## 1. Entry Point

**Location**: `.venv/lib/python3.11/site-packages/torch/_tensor.py:~1060`

When you call `tensor.backward()`, it's defined in the `Tensor` class:

```python
def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    r"""Computes the gradient of current tensor wrt graph leaves.

    The graph is differentiated using the chain rule. If the tensor is
    non-scalar (i.e. its data has more than one element) and requires
    gradient, the function additionally requires specifying a ``gradient``.
    """
    if has_torch_function_unary(self):
        return handle_torch_function(
            Tensor.backward,
            (self,),
            self,
            gradient=gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs,
        )
    torch.autograd.backward(
        self, gradient, retain_graph, create_graph, inputs=inputs
    )
```

## 2. Python Autograd Layer

**Location**: `.venv/lib/python3.11/site-packages/torch/autograd/__init__.py:~170`

The `torch.autograd.backward()` function handles:
- Input validation and conversion to tuples
- Creating gradient tensors if not provided
- Calling the C++ engine

```python
def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
) -> None:
    # ... validation and setup ...

    _engine_run_backward(
        tensors,
        grad_tensors_,
        retain_graph,
        create_graph,
        inputs_tuple,
        allow_unreachable=True,
        accumulate_grad=True,
    )
```

## 3. Engine Bridge

**Location**: `.venv/lib/python3.11/site-packages/torch/autograd/graph.py`

The bridge between Python and C++:

```python
def _engine_run_backward(
    t_outputs: Sequence[Union[torch.Tensor, GradientEdge]],
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    if attach_logging_hooks:
        unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    try:
        return Variable._execution_engine.run_backward(  # Calls into the C++ engine
            t_outputs, *args, **kwargs
        )
    finally:
        if attach_logging_hooks:
            unregister_hooks()
```

## 4. Variable and Execution Engine

**Location**: `.venv/lib/python3.11/site-packages/torch/autograd/variable.py`

```python
from torch._C import _ImperativeEngine as ImperativeEngine

class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):
    _execution_engine = ImperativeEngine()  # Singleton C++ engine
```

The `_execution_engine` is defined in C++ (`torch/csrc/autograd/python_engine.cpp`).

## 5. C++ Execution Engine

**Location**: `.venv/lib/python3.11/site-packages/torch/include/torch/csrc/autograd/engine.h`

The C++ engine header shows the core architecture:

```cpp
// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

struct TORCH_API Engine {
  // Given a list of (Node, input number) pairs computes the value of the graph
  // by following next_edge references.
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      bool accumulate_grad,
      const edge_list& outputs = {});

  void evaluate_function(
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs,
      const std::shared_ptr<ReadyQueue>& cpu_ready_queue);
};
```

### Key Features:
- **Topological traversal** of the computational graph
- **Priority queue** (`ReadyQueue`) to schedule node execution
- **NodeTask** objects representing gradient computations
- Support for **multithreading** and **reentrant backward passes**
- Maximum reentrant depth of 60 to avoid deadlocks

## 6. Computational Graph Structure

### Nodes
**Location**: `.venv/lib/python3.11/site-packages/torch/include/torch/csrc/autograd/function.h`

A `Node` represents an operation in the autograd graph:

```cpp
// A `Node` is an abstract class that represents an operation taking zero
// or more input `Variable`s and producing zero or more output `Variable`s.
struct TORCH_API Node : std::enable_shared_from_this<Node> {
  // The most important method - computes gradients
  virtual variable_list apply(variable_list&& inputs) = 0;

  // Edges to the next nodes in the graph
  edge_list next_edges_;

  // Sequence number for topological ordering
  uint64_t sequence_nr_;
};
```

Examples of concrete Node subclasses:
- `AddBackward` - gradient of addition
- `MulBackward` - gradient of multiplication
- `AccumulateGrad` - accumulates gradients into leaf tensors
- `GraphRoot` - starting point for backward pass

### Edges

Edges connect nodes and are represented as `(Node, input_nr)` pairs. When multiple edges point to the same input, gradients are implicitly summed.

### grad_fn

Each tensor created by an operation stores a reference to the `Node` that created it:

```python
>>> x = torch.tensor([1.0, 2.0], requires_grad=True)
>>> y = x * 2
>>> y.grad_fn
<MulBackward0 object at 0x...>
>>> y.grad_fn.next_functions
((AccumulateGrad object at 0x..., 0),)
```

## 7. Backward Execution Flow

```
User calls: tensor.backward()
    ↓
torch._tensor.Tensor.backward()
    ↓
torch.autograd.backward()
    ↓
torch.autograd.graph._engine_run_backward()
    ↓
Variable._execution_engine.run_backward() [Python → C++ boundary]
    ↓
Engine::execute() [C++]
    ↓
Topological traversal of computation graph:
    ├── Pop NodeTask from ReadyQueue
    ├── Call Node::apply() to compute local gradients
    ├── Apply chain rule: multiply incoming gradients × local gradients
    ├── Push results to next nodes via next_edges
    └── Accumulate gradients at leaf tensors (AccumulateGrad nodes)
```

## 8. Chain Rule Implementation

The engine implements the **chain rule** through graph traversal:

1. **Start**: Begin at output tensors (loss)
2. **Traverse**: Visit nodes in reverse topological order
3. **Compute**: At each node, call `Node::apply(incoming_gradients)` to compute local gradients
4. **Multiply**: Combine incoming gradients with local gradients (chain rule)
5. **Propagate**: Send results to previous nodes via `next_edges`
6. **Accumulate**: At leaf tensors, accumulate gradients into `.grad` attribute

### Example:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3        # Creates MulBackward0 node
z = y + 5        # Creates AddBackward0 node
z.backward()     # Triggers backward pass
```

**Graph structure**:
```
z (AddBackward0) → y (MulBackward0) → x (AccumulateGrad)
```

**Backward pass**:
1. Start at `z` with gradient = 1.0
2. `AddBackward0.apply([1.0])` → returns `[1.0]` (derivative of +5)
3. Pass to `MulBackward0.apply([1.0])` → returns `[3.0]` (derivative of *3)
4. `AccumulateGrad` stores 3.0 in `x.grad`

## 9. Custom Functions

**Location**: `.venv/lib/python3.11/site-packages/torch/autograd/function.py`

Users can define custom backward operations:

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output * y
        grad_y = grad_output * x
        return grad_x, grad_y
```

The `FunctionCtx` provides:
- `save_for_backward()` - save tensors for backward pass
- `saved_tensors` - retrieve saved tensors
- Custom attributes for non-tensor data

## 10. Key Concepts

### Leaf Tensors
Tensors created directly by the user with `requires_grad=True`. They have:
- `grad_fn = None`
- `.grad` attribute that accumulates gradients
- `is_leaf = True`

### Non-Leaf Tensors
Tensors created by operations. They have:
- `grad_fn != None` (points to the operation that created them)
- `.grad = None` by default (unless `retain_grad()` is called)
- `is_leaf = False`

### Graph Retention
- By default, the graph is freed after `backward()` to save memory
- `retain_graph=True` keeps the graph for multiple backward passes
- `create_graph=True` builds a graph of the derivative for higher-order gradients

## 11. Important Source Files

| File | Purpose |
|------|---------|
| `torch/_tensor.py` | Tensor class with `backward()` method |
| `torch/autograd/__init__.py` | Main `backward()` function and gradient computation |
| `torch/autograd/variable.py` | Variable class with `_execution_engine` singleton |
| `torch/autograd/graph.py` | Graph traversal and `_engine_run_backward()` |
| `torch/autograd/function.py` | Custom Function API and FunctionCtx |
| `torch/include/torch/csrc/autograd/engine.h` | C++ Engine class definition |
| `torch/include/torch/csrc/autograd/function.h` | C++ Node class definition |

## 12. Performance Optimizations

The C++ engine includes several optimizations:
- **Parallel execution**: Multiple independent nodes can execute concurrently
- **Thread pool**: Reuses threads for evaluating nodes
- **Priority queue**: Processes nodes in optimal order based on sequence numbers
- **Memory efficiency**: Frees intermediate results when no longer needed
- **Reentrant support**: Handles nested backward calls (up to depth 60)

## Summary

PyTorch's `backward()` is a sophisticated system that bridges Python and C++:
- **Python layer**: Provides user-friendly API and validation
- **C++ engine**: Handles performance-critical graph traversal and gradient computation
- **Dynamic graphs**: Built on-the-fly during forward pass
- **Chain rule**: Automatically applied through topological graph traversal
- **Extensible**: Users can define custom operations with custom gradients

The architecture cleanly separates concerns while maintaining high performance for deep learning workloads.
