# PyTorch's forward() Architecture

This document explains how PyTorch's `forward()` method works internally, based on investigation of the PyTorch source code in `.venv`.

## Overview

PyTorch's forward pass is where:
1. **Input data flows through the neural network** layers to produce predictions
2. **Computational graph is built dynamically** for automatic differentiation
3. **Hooks are executed** before and after the forward computation
4. **Backward nodes are created** that will later compute gradients

Unlike `backward()` which traverses an existing graph, `forward()` **builds** the graph as operations execute.

## 1. Entry Point: The `__call__` Mechanism

**Location**: `.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:~1914`

When you call a PyTorch module like `model(inputs)`, Python invokes the `__call__` method:

```python
__call__: Callable[..., Any] = _wrapped_call_impl
```

**Key insight**: You never call `forward()` directly. The `__call__` mechanism handles hooks, compilation, and orchestration before/after calling `forward()`.

## 2. The Call Stack

```
User code: predictions = model(inputs)
    ↓
Module.__call__(*args, **kwargs)
    ↓
Module._wrapped_call_impl(*args, **kwargs)
    ↓
Module._call_impl(*args, **kwargs)
    ↓
[Execute forward pre-hooks]
    ↓
self.forward(*args, **kwargs)  ← Your custom forward implementation
    ↓
[Execute forward post-hooks]
    ↓
[Setup backward hooks]
    ↓
Return result
```

## 3. The `_wrapped_call_impl` Method

**Location**: `.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:~1771`

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)  # Compiled path
    else:
        return self._call_impl(*args, **kwargs)  # Normal path
```

This wrapper checks if the module has been compiled (with `torch.compile()`). If so, it uses the compiled version for performance.

## 4. The Core `_call_impl` Method

**Location**: `.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:~1779`

This is where the main orchestration happens:

```python
def _call_impl(self, *args, **kwargs):
    # Choose forward implementation (normal or tracing mode)
    forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)

    # Fast path: if no hooks, just call forward directly
    if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks
            or self._forward_pre_hooks or _global_backward_pre_hooks
            or _global_backward_hooks or _global_forward_hooks
            or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)

    # Slow path: execute hooks and forward together
    # [Detailed hook execution code...]

    # Execute forward pre-hooks
    for hook in forward_pre_hooks:
        result = hook(self, args)
        if result is not None:
            args = result

    # Execute the actual forward method
    result = forward_call(*args, **kwargs)

    # Execute forward post-hooks
    for hook in forward_hooks:
        hook_result = hook(self, args, result)
        if hook_result is not None:
            result = hook_result

    return result
```

### Key Features:
- **Fast path optimization**: If no hooks registered, directly calls `forward()`
- **Tracing support**: Uses `_slow_forward` during JIT tracing
- **Hook execution**: Pre-hooks can modify inputs, post-hooks can modify outputs
- **Backward hook setup**: Registers hooks for the backward pass

## 5. The User-Defined `forward()` Method

**Location**: Your code (e.g., `sample.py:20-23`)

```python
def forward(self, x):
    x = F.relu(self.fc1(x))  # First layer + activation
    x = self.fc2(x)          # Second layer
    return x
```

This is where you define what happens to the input data:
- Apply transformations (linear layers, convolutions, etc.)
- Apply activations (ReLU, sigmoid, etc.)
- Return output

**Important**: As operations execute, PyTorch automatically builds the computational graph for backpropagation.

## 6. Computational Graph Building

### How It Works

Every tensor operation during forward pass creates **backward nodes** that will later compute gradients.

**Location**: `.venv/lib/python3.11/site-packages/torch/include/torch/csrc/autograd/variable.h:~205`

Each tensor has an `AutogradMeta` struct containing:

```cpp
struct AutogradMeta {
  Variable grad_;                        // Accumulated gradient
  std::shared_ptr<Node> grad_fn_;        // Function that created this tensor
  std::weak_ptr<Node> grad_accumulator_; // For leaf tensors
  bool requires_grad_{false};            // Track gradients?
  uint32_t output_nr_;                   // Output index from the operation
  // ... more fields ...
};
```

### Graph Construction Example

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3        # Creates MulBackward0 node
z = y + 5        # Creates AddBackward0 node
```

**What happens internally**:

1. **`x * 3` executes**:
   - Creates a new tensor `y`
   - Creates a `MulBackward0` node
   - Sets `y.grad_fn = MulBackward0`
   - `MulBackward0.next_edges` points to `x`'s AccumulateGrad

2. **`y + 5` executes**:
   - Creates a new tensor `z`
   - Creates an `AddBackward0` node
   - Sets `z.grad_fn = AddBackward0`
   - `AddBackward0.next_edges` points to `y.grad_fn` (MulBackward0)

**Graph structure**:
```
z (output)
  └── grad_fn: AddBackward0
        └── next_edge: MulBackward0
              └── next_edge: AccumulateGrad (for x)
```

## 7. Backward Node Definitions

**Location**: `.venv/lib/python3.11/site-packages/torch/include/torch/csrc/autograd/generated/Functions.h`

Each operation has a corresponding backward node class:

```cpp
struct TORCH_API AddBackward0 : public TraceableFunction {
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }

  // Saved data needed for backward pass
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;
};

struct TORCH_API MulBackward0 : public TraceableFunction {
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward0"; }

  SavedVariable other_;  // The other operand (saved for gradient computation)
  SavedVariable self_;
};
```

### Key Methods:
- **`apply(grads)`**: Computes local gradients (called during backward pass)
- **`release_variables()`**: Frees saved tensors to save memory
- **Saved variables**: Store input/output data needed to compute gradients later

## 8. Operation Dispatch and Graph Construction

When you call operations like `+`, `*`, `relu()`, etc., PyTorch:

1. **Checks if gradient tracking is needed**:
   - Is `torch.grad_enabled()` true?
   - Does any input have `requires_grad=True`?

2. **Executes the forward computation**:
   - Performs the actual mathematical operation
   - Creates output tensor

3. **Creates and attaches backward node** (if gradients needed):
   - Instantiates appropriate `*Backward0` object
   - Saves necessary tensors/metadata for backward pass
   - Sets output tensor's `grad_fn` to this node
   - Links node to input tensors via `next_edges`

### Example: ReLU Operation

```python
y = F.relu(x)  # x has requires_grad=True
```

**Internally**:
1. Execute ReLU: `y_data = max(0, x_data)`
2. Create `ReluBackward0` node
3. Save `x` in the node (needed for gradient: `grad * (x > 0)`)
4. Set `y.grad_fn = ReluBackward0`
5. Link `ReluBackward0.next_edges[0] = (x.grad_fn, 0)`

## 9. The Hooks System

PyTorch supports four types of hooks on modules:

### Forward Pre-Hooks

**Location**: `.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:~1612`

```python
def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
    """
    Hook signature: hook(module, args) -> None or modified args
    Called BEFORE forward() is invoked
    Can modify the input arguments
    """
```

**Use cases**:
- Input validation
- Input preprocessing
- Debugging/logging

### Forward Hooks

**Location**: `.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:~1678`

```python
def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
    """
    Hook signature: hook(module, args, output) -> None or modified output
    Called AFTER forward() computes output
    Can modify the output
    """
```

**Use cases**:
- Feature extraction
- Output logging
- Activation visualization

### Backward Pre-Hooks

```python
def register_full_backward_pre_hook(self, hook, *, prepend=False):
    """
    Hook signature: hook(module, grad_output) -> None or modified grad_output
    Called BEFORE backward() on this module
    """
```

### Backward Hooks

```python
def register_full_backward_hook(self, hook, *, prepend=False):
    """
    Hook signature: hook(module, grad_input, grad_output) -> None or modified grad_input
    Called AFTER backward() on this module
    """
```

### Global vs Module-Specific Hooks

**Global hooks** (module.py:~113-119):
```python
_global_forward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks: dict[int, Callable] = OrderedDict()
_global_backward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_backward_hooks: dict[int, Callable] = OrderedDict()
```

These hooks apply to **all** modules globally, useful for:
- Profiling entire networks
- Global debugging
- Instrumentation frameworks

## 10. Forward Execution Flow in Your Sample Code

From `sample.py`:

```python
model = SimpleNN(input_size, hidden_size, output_size)
predictions = model(inputs)  # inputs: [32, 3072]
```

**Step-by-step execution**:

1. **`model(inputs)`** → Calls `Module.__call__`

2. **`__call__`** → Calls `_wrapped_call_impl(inputs)`

3. **`_wrapped_call_impl`** → Calls `_call_impl(inputs)`

4. **`_call_impl`** determines:
   - No hooks registered → Fast path
   - Calls `self.forward(inputs)` directly

5. **`SimpleNN.forward(inputs)`** executes:
   ```python
   def forward(self, x):                    # x: [32, 3072]
       x = F.relu(self.fc1(x))              # Step A
       x = self.fc2(x)                      # Step B
       return x                             # x: [32, 10]
   ```

6. **Step A: `F.relu(self.fc1(x))`**:

   a. **`self.fc1(x)`** (Linear layer):
      - Computes: `output = x @ weight.T + bias`
      - Creates `AddmmBackward0` node
      - Output shape: `[32, 128]`
      - Sets `output.grad_fn = AddmmBackward0`

   b. **`F.relu(output)`**:
      - Computes: `output2 = max(0, output)`
      - Creates `ReluBackward0` node
      - Output shape: `[32, 128]`
      - Sets `output2.grad_fn = ReluBackward0`
      - `ReluBackward0.next_edges[0]` → `AddmmBackward0`

7. **Step B: `self.fc2(x)`**:
   - Computes: `predictions = x @ weight2.T + bias2`
   - Creates `AddmmBackward0` node
   - Output shape: `[32, 10]`
   - Sets `predictions.grad_fn = AddmmBackward0`
   - This node links back to `ReluBackward0`

8. **Return predictions** with complete computational graph

**Final graph structure**:
```
predictions [32, 10]
  └── grad_fn: AddmmBackward0 (fc2)
        └── next_edge: ReluBackward0
              └── next_edge: AddmmBackward0 (fc1)
                    ├── next_edge: AccumulateGrad (fc1.weight)
                    └── next_edge: AccumulateGrad (fc1.bias)
```

## 11. Key Concepts

### Leaf Tensors
Tensors created directly by the user (parameters, inputs with `requires_grad=True`):
- `grad_fn = None`
- `is_leaf = True`
- Can accumulate gradients in `.grad`

```python
>>> model.fc1.weight.is_leaf
True
>>> model.fc1.weight.grad_fn
None
```

### Non-Leaf Tensors
Tensors created by operations:
- `grad_fn != None`
- `is_leaf = False`
- `.grad` is `None` by default (unless `retain_grad()` called)

```python
>>> predictions.is_leaf
False
>>> predictions.grad_fn
<AddmmBackward0 object at 0x...>
```

### Dynamic Computational Graphs
PyTorch builds graphs **dynamically** during forward pass:
- Different forward passes can create different graphs
- Enables control flow (if/else, loops) in models
- Graph is destroyed after backward unless `retain_graph=True`

### Gradient Mode
Controls whether graph building happens:

```python
>>> torch.is_grad_enabled()
True

>>> with torch.no_grad():
...     y = model(x)  # No graph built, y.grad_fn = None

>>> @torch.inference_mode()  # Even faster than no_grad
... def predict(x):
...     return model(x)
```

## 12. Tensor Operations and Backward Nodes

Common operations and their backward nodes:

| Operation | Backward Node | Saved Data |
|-----------|---------------|------------|
| `a + b` | `AddBackward0` | Scalar types, alpha |
| `a * b` | `MulBackward0` | `a`, `b` |
| `a @ b` | `MmBackward0` | `a`, `b` |
| `a.sum()` | `SumBackward0` | `a.shape` |
| `a.mean()` | `MeanBackward0` | `a.shape`, `a.numel()` |
| `relu(a)` | `ReluBackward0` | `a` (or just `a > 0`) |
| `sigmoid(a)` | `SigmoidBackward0` | Result (output) |
| `a.view(shape)` | `ViewBackward0` | Original shape |
| `Linear(x)` | `AddmmBackward0` | `x`, `weight` |
| `Conv2d(x)` | `ConvolutionBackward0` | `x`, `weight`, parameters |

## 13. Important Source Files

| File | Purpose |
|------|---------|
| `torch/nn/modules/module.py` | `Module` class, `__call__`, hooks system |
| `torch/nn/modules/module.py:~1914` | `__call__ = _wrapped_call_impl` assignment |
| `torch/nn/modules/module.py:~1779` | `_call_impl()` - main orchestration |
| `torch/include/torch/csrc/autograd/variable.h` | `AutogradMeta` struct, `grad_fn` storage |
| `torch/include/torch/csrc/autograd/function.h` | `Node` base class for all backward nodes |
| `torch/include/torch/csrc/autograd/generated/Functions.h` | All backward node definitions (`AddBackward0`, etc.) |
| `torch/autograd/function.py` | Python `Function` class for custom operations |
| `torch/nn/functional.py` | Functional operations (relu, etc.) |

## 14. Custom Operations with `torch.autograd.Function`

**Location**: `.venv/lib/python3.11/site-packages/torch/autograd/function.py`

You can define custom operations with custom gradients:

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: compute output and save data for backward
        ctx: context object for saving data
        """
        ctx.save_for_backward(x)  # Save input for gradient computation
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient
        grad_output: gradient flowing from next layer
        Returns: gradient with respect to input
        """
        x, = ctx.saved_tensors
        grad_x = grad_output * 2 * x  # d(x²)/dx = 2x
        return grad_x

# Usage
x = torch.tensor([3.0], requires_grad=True)
y = MySquare.apply(x)  # y = 9.0
y.backward()
print(x.grad)  # tensor([6.0]) = 2 * 3
```

**What happens**:
1. `MySquare.apply(x)` creates a custom backward node
2. During forward: saves `x`, computes and returns `x²`
3. During backward: retrieves `x`, computes `grad_output * 2x`

## 15. Performance Optimizations

### Fast Path (No Hooks)
If no hooks are registered, `_call_impl` directly calls `forward()` without overhead.

### Compiled Models
`torch.compile()` can compile the forward pass for significant speedups:
```python
model = torch.compile(model)  # Uses _compiled_call_impl
```

### Inference Mode
Disables view tracking and version counter:
```python
with torch.inference_mode():
    predictions = model(inputs)  # Faster than no_grad()
```

### Graph Caching
For static graphs (same architecture every forward), PyTorch can optimize subsequent passes.

## 16. Differences: forward() vs backward()

| Aspect | forward() | backward() |
|--------|-----------|------------|
| **Purpose** | Compute predictions | Compute gradients |
| **Direction** | Input → Output | Output → Input |
| **Graph** | **Builds** graph | **Traverses** existing graph |
| **User code** | User defines logic | PyTorch auto-generates |
| **Entry** | `model(input)` → `__call__` → `forward()` | `loss.backward()` |
| **Operations** | Mathematical ops | Gradient computation (chain rule) |
| **Hooks** | Forward pre/post hooks | Backward pre/post hooks |
| **Output** | Predictions (tensors) | Gradients (stored in `.grad`) |
| **Speed** | Fast path if no hooks | Complex graph traversal |

## 17. Complete Execution Example

```python
# Setup
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Forward pass
y = x * w      # Creates MulBackward0, saves x and w
z = y + b      # Creates AddBackward0
loss = z ** 2  # Creates PowBackward0

# Computational graph built:
# loss (PowBackward0) → z (AddBackward0) → y (MulBackward0) → x, w (AccumulateGrad)

# Check graph
print(loss.grad_fn)  # <PowBackward0>
print(loss.grad_fn.next_functions)  # ((AddBackward0, 0),)
print(z.grad_fn)  # <AddBackward0>
print(z.grad_fn.next_functions)  # ((MulBackward0, 0), (AccumulateGrad, 0))

# Backward pass (traverses the graph)
loss.backward()

# Gradients computed and accumulated
print(x.grad)  # d(loss)/dx = 2*z*w = 2*7*3 = 42
print(w.grad)  # d(loss)/dw = 2*z*x = 2*7*2 = 28
print(b.grad)  # d(loss)/db = 2*z*1 = 2*7 = 14
```

## Summary

PyTorch's `forward()` is the **graph construction** phase of automatic differentiation:

1. **Entry**: `model(inputs)` → `__call__` → `_wrapped_call_impl` → `_call_impl` → `forward()`

2. **User code**: You define `forward()` to specify how inputs transform to outputs

3. **Graph building**: As operations execute, PyTorch:
   - Creates backward nodes (`AddBackward0`, `MulBackward0`, etc.)
   - Stores necessary data for gradient computation
   - Links nodes via `next_edges` to build graph

4. **Hooks**: Pre/post hooks can modify inputs/outputs and enable instrumentation

5. **Result**: Returns predictions **and** a computational graph ready for `backward()`

6. **Philosophy**:
   - `forward()` = **define-by-run** (dynamic graph construction)
   - `backward()` = **automatic differentiation** (graph traversal)

The architecture cleanly separates user-facing simplicity (`forward()` is just Python code) from internal complexity (automatic graph building and differentiation), making PyTorch both powerful and intuitive.
