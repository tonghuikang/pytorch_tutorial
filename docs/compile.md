# PyTorch's torch.compile() Architecture

This document explains how PyTorch's `torch.compile()` works internally, based on investigation of the PyTorch source code in `.venv`.

## Overview

`torch.compile()` is PyTorch's JIT (Just-In-Time) compiler introduced in PyTorch 2.0 that optimizes models for faster execution. Unlike eager mode where operations execute immediately, `torch.compile()`:

1. **Captures computational graphs** dynamically using TorchDynamo
2. **Compiles graphs** using backends like TorchInductor
3. **Caches compiled code** for reuse
4. **Falls back to eager** mode when necessary

**Key Innovation**: Compile models written in normal PyTorch code without changing the programming model.

## 1. Entry Point

**Location**: `.venv/lib/python3.11/site-packages/torch/__init__.py:~2478`

```python
def compile(
    model: Optional[Callable] = None,
    *,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    backend: Union[str, Callable] = "inductor",
    mode: Union[str, None] = None,
    options: Optional[dict] = None,
    disable: bool = False,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    """
    # ... validation and setup ...

    if backend == "inductor":
        backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    else:
        backend = _TorchCompileWrapper(backend, mode, options, dynamic)

    return torch._dynamo.optimize(
        backend=backend,
        nopython=fullgraph,
        dynamic=dynamic,
        disable=disable,
        guard_filter_fn=guard_filter_fn,
    )(model)
```

### Key Parameters

- **backend**: Which compiler to use (default: "inductor")
  - `"inductor"`: Default, Triton-based codegen for GPUs
  - `"aot_eager"`: AOT Autograd with eager backend
  - `"cudagraphs"`: CUDA graphs for reduced overhead
  - Custom callable backends

- **fullgraph**: If `True`, require single graph (error on graph breaks)
- **dynamic**: Dynamic shape handling
  - `True`: Compile for dynamic shapes upfront
  - `False`: Always specialize to specific shapes
  - `None`: Auto-detect and recompile as needed (default)

- **mode**: Optimization mode
  - `"default"`: Balanced performance/overhead
  - `"reduce-overhead"`: Use CUDA graphs, minimize Python overhead
  - `"max-autotune"`: Aggressive optimization, profile matmuls
  - `"max-autotune-no-cudagraphs"`: Max autotune without CUDA graphs

## 2. Architecture Overview

```
User Code: model = torch.compile(my_model)
              ↓
torch.compile() (torch/__init__.py)
              ↓
torch._dynamo.optimize() (torch/_dynamo/eval_frame.py)
              ↓
OptimizeContext wraps the model
              ↓
On model call: Python frame evaluation hook triggers
              ↓
TorchDynamo: Bytecode analysis & graph capture
              ↓
              ├─→ [Graph Break] → Continue in eager mode
              │
              └─→ [Successful Capture] → FX Graph
                          ↓
              Backend Compilation (e.g., Inductor)
                          ↓
              Optimized Code (Triton kernels, fused ops)
                          ↓
              Cache compiled function
                          ↓
              Execute compiled code
```

## 3. TorchDynamo: The Graph Capture System

**Location**: `.venv/lib/python3.11/site-packages/torch/_dynamo/`

TorchDynamo is the core technology that enables `torch.compile()` by capturing PyTorch programs into FX graphs.

### 3.1 How TorchDynamo Works

**Frame Evaluation Hook**: TorchDynamo intercepts Python bytecode execution using CPython's frame evaluation API:

```python
# In eval_frame.py:~1251
def optimize(*args, **kwargs) -> Union[OptimizeContext, _NullDecorator]:
    return _optimize(rebuild_ctx, *args, **kwargs)

def _optimize(
    rebuild_ctx,
    backend="inductor",
    *,
    nopython=False,
    dynamic=None,
    ...
):
    # Install frame evaluation callback
    return _optimize_catch_errors(
        convert_frame.convert_frame(backend, hooks, package=package),
        hooks,
        backend_ctx_ctor,
        ...
    )
```

**What happens when compiled function is called**:

1. **Python starts executing function** → Enters first frame
2. **Frame evaluation hook fires** → TorchDynamo intercepts
3. **Bytecode analysis** → Parse Python bytecode instructions
4. **Symbolic tracing** → Track tensor operations and control flow
5. **Graph construction** → Build FX graph from operations
6. **Guard generation** → Create guards (e.g., tensor shapes, dtypes)
7. **Backend compilation** → Pass graph to compiler backend
8. **Code caching** → Cache compiled result on code object
9. **Execution** → Run optimized code

### 3.2 Bytecode Analysis

**Location**: `.venv/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py`

TorchDynamo analyzes Python bytecode instruction by instruction:

```python
# Example bytecode for: y = x + 1
LOAD_FAST      x
LOAD_CONST     1
BINARY_ADD
STORE_FAST     y
```

For each instruction, TorchDynamo:
- Tracks variable values symbolically
- Records tensor operations
- Handles control flow (if/else, loops, etc.)
- Detects graph breaks

### 3.3 Graph Breaks

**Graph Break** = Point where TorchDynamo cannot continue tracing

Common causes:
- Unsupported Python features
- Dynamic control flow that can't be captured
- Calls to non-traceable functions
- Data-dependent operations

**Example**:
```python
@torch.compile
def f(x, flag):
    y = x + 1
    if flag:  # Potential graph break if flag varies
        y = y * 2
    return y
```

When a graph break occurs:
1. Compile the graph captured so far
2. Fall back to eager mode for unsupported section
3. Resume graph capture after the break
4. Result: Multiple graph segments

**Debugging**: Use `fullgraph=True` to error on graph breaks:
```python
@torch.compile(fullgraph=True)
def f(x):
    return x + 1  # Must be fully capturable
```

## 4. Backends

### 4.1 TorchInductor (Default)

**Location**: `.venv/lib/python3.11/site-packages/torch/_inductor/`

TorchInductor is the default backend that generates optimized code:

**What it does**:
1. **Takes FX Graph** from TorchDynamo
2. **Applies graph optimizations**:
   - Operator fusion (combine multiple ops)
   - Memory planning
   - Layout optimization
3. **Code generation**:
   - **GPU**: Generates Triton kernels
   - **CPU**: Generates C++/OpenMP code
4. **Returns compiled function**

**Key Features**:
- **Fusion**: Combines pointwise operations (e.g., `add + relu + mul`)
- **Triton**: GPU code generation using Triton language
- **Auto-tuning**: Profiles multiple implementations, picks fastest
- **Memory optimization**: Reduces memory allocations

**Example optimization**:
```python
# Original
y = x + 1
z = torch.relu(y)
w = z * 2

# Inductor fuses into single kernel:
w = fused_kernel(x)  # (x + 1).relu() * 2 in one pass
```

### 4.2 Other Backends

Available backends (via `torch._dynamo.list_backends()`):

- **`aot_eager`**: AOT Autograd + eager execution (testing)
- **`cudagraphs`**: CUDA graphs for reduced Python overhead
- **`ipex`**: Intel Extension for PyTorch
- **`onnxrt`**: ONNX Runtime
- **`tensorrt`**: NVIDIA TensorRT
- **`tvm`**: Apache TVM
- **Custom**: User-defined compiler functions

**Custom Backend Example**:
```python
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    gm: FX GraphModule representing captured computation
    example_inputs: Example inputs for shape/dtype inference
    Returns: Callable that runs optimized graph
    """
    print(f"Compiling graph with {len(gm.graph.nodes)} nodes")
    # ... custom compilation logic ...
    return gm.forward  # Or optimized version

model = torch.compile(model, backend=my_backend)
```

## 5. Guards System

**Location**: `.venv/lib/python3.11/site-packages/torch/_guards.py`

Guards ensure compiled code is only used when assumptions hold.

### 5.1 What Are Guards?

Guards are runtime checks that verify assumptions made during compilation:

```python
# Example guards for: def f(x): return x + 1
- tensor 'x' has dtype torch.float32
- tensor 'x' has shape [32, 128]
- tensor 'x' requires_grad == True
- tensor 'x' is on device 'cuda:0'
```

If guards fail → **Guard Failure** → Recompile with new assumptions

### 5.2 Guard Evaluation Flow

```
Execute compiled function
    ↓
Check guards
    ↓
    ├─→ [Guards Pass] → Execute cached compiled code
    │
    └─→ [Guards Fail] → Recompile or fall back to eager
```

### 5.3 Recompilation Limits

**Location**: `torch._dynamo.config.recompile_limit` (default: 8)

If a code object gets recompiled >8 times (too many guard failures):
- Fall back to eager execution permanently
- Warning: "TorchDynamo hit recompile limit"

**Debugging guards**:
```bash
TORCH_LOGS=guards python script.py
```

## 6. Compilation Workflow Example

Let's trace what happens when you compile a simple function:

```python
import torch

@torch.compile
def add_relu(x):
    y = x + 1
    z = torch.relu(y)
    return z

# First call
x = torch.randn(32, 128, device='cuda')
result = add_relu(x)
```

**Step-by-step execution**:

### Step 1: `@torch.compile` decorator application
```python
# Transforms to:
add_relu = torch.compile(add_relu)
# Returns OptimizeContext wrapping add_relu
```

### Step 2: First call to `add_relu(x)`
```python
# OptimizeContext.__call__ is invoked
# Frame evaluation hook installed
```

### Step 3: TorchDynamo intercepts bytecode
```python
# Bytecode of add_relu:
LOAD_FAST       x
LOAD_CONST      1
BINARY_ADD      # y = x + 1
STORE_FAST      y
LOAD_GLOBAL     torch
LOAD_ATTR       relu
LOAD_FAST       y
CALL_FUNCTION   # z = torch.relu(y)
STORE_FAST      z
LOAD_FAST       z
RETURN_VALUE    # return z
```

### Step 4: Symbolic execution & graph building
```python
# TorchDynamo builds FX Graph:
graph():
    %x : torch.Tensor [#users=1]
    %add_result : torch.Tensor = call_function[target=torch.ops.aten.add.Tensor](
        args=(%x, 1), kwargs={}
    )
    %relu_result : torch.Tensor = call_function[target=torch.ops.aten.relu.default](
        args=(%add_result,), kwargs={}
    )
    return relu_result
```

### Step 5: Guard generation
```python
# Generated guards:
- x.dtype == torch.float32
- x.device == device(type='cuda', index=0)
- x.ndim == 2
- x.requires_grad == False
# (dynamic=None, so shapes are specialized)
- x.shape[0] == 32
- x.shape[1] == 128
```

### Step 6: Backend compilation (Inductor)
```python
# Inductor receives FX graph
# Analyzes: add + relu can be fused
# Generates Triton kernel:

@triton.jit
def fused_add_relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = x + 1.0
    z = tl.maximum(y, 0.0)  # relu
    tl.store(out_ptr + offs, z, mask=mask)
```

### Step 7: Caching
```python
# Compiled function cached on code object:
add_relu.__code__._torchdynamo_cache[guard_hash] = compiled_fn
```

### Step 8: Execution
```python
# Run optimized Triton kernel
result = compiled_fn(x)
```

### Step 9: Subsequent calls
```python
# Second call with same shape:
x2 = torch.randn(32, 128, device='cuda')
result2 = add_relu(x2)

# Flow:
1. Check guards → Pass (same dtype, device, shape)
2. Retrieve cached compiled_fn
3. Execute → Fast!
```

### Step 10: Shape change
```python
# Call with different shape:
x3 = torch.randn(64, 256, device='cuda')  # Different shape!
result3 = add_relu(x3)

# Flow:
1. Check guards → Fail (shape mismatch: [64, 256] != [32, 128])
2. Recompile with new guards (32→64, 128→256)
3. Cache new compiled version
4. Execute

# Now have 2 cached versions:
# - add_relu[shape=(32, 128)]
# - add_relu[shape=(64, 256)]
```

## 7. Dynamic Shapes

### 7.1 The Problem

By default, `torch.compile()` specializes to specific shapes:

```python
@torch.compile
def f(x):
    return x + 1

f(torch.randn(32, 128))   # Compiles for [32, 128]
f(torch.randn(64, 128))   # Recompiles for [64, 128]
f(torch.randn(128, 128))  # Recompiles for [128, 128]
# Result: Many recompilations!
```

### 7.2 The Solution: `dynamic=True`

```python
@torch.compile(dynamic=True)
def f(x):
    return x + 1

f(torch.randn(32, 128))   # Compiles for [s0, 128] (symbolic size s0)
f(torch.randn(64, 128))   # Reuses! s0=64
f(torch.randn(128, 128))  # Reuses! s0=128
```

**How it works**:
- Introduces symbolic sizes (`s0`, `s1`, etc.)
- Generates more general code
- Some optimizations may not apply
- Trade-off: Fewer recompiles vs potentially less optimal code

### 7.3 Automatic Dynamic Shapes (`dynamic=None`)

Default behavior: Auto-detect when dynamism is needed

1. **First call**: Specialize to concrete shapes
2. **Shape changes**: Detect, recompile with symbolic shapes
3. **Best of both worlds**: Optimal for static, dynamic when needed

## 8. Compiled Autograd

**Location**: `.venv/lib/python3.11/site-packages/torch/_dynamo/compiled_autograd.py`

PyTorch can also compile the backward pass!

```python
torch._dynamo.config.compiled_autograd = True

@torch.compile
def f(x):
    return (x ** 2).sum()

x = torch.randn(1000, requires_grad=True)
y = f(x)      # Forward: compiled
y.backward()  # Backward: also compiled!
```

**Benefits**:
- Faster gradient computation
- Fused backward operations
- Reduced Python overhead in training

## 9. Debugging and Profiling

### 9.1 Explain Mode

See what `torch.compile()` captured:

```python
explanation = torch._dynamo.explain(add_relu)(x)
print(explanation.graph_count)      # Number of graphs
print(explanation.graph_break_count)  # Number of breaks
print(explanation.graphs[0].code)   # FX graph code
```

### 9.2 Environment Variables

```bash
# See all graph breaks
TORCH_LOGS=graph_breaks python script.py

# See guards
TORCH_LOGS=guards python script.py

# See dynamic shape decisions
TORCH_LOGS=dynamic python script.py

# See performance hints
TORCH_LOGS=perf_hints python script.py

# See recompilations
TORCH_LOGS=recompiles python script.py

# Enable all
TORCH_LOGS="+all" python script.py
```

### 9.3 Inspecting Compiled Code

```python
@torch.compile
def f(x):
    return x + 1

# See generated Triton code
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.graph_diagram = True

f(torch.randn(32, 128, device='cuda'))
# Outputs Triton kernel source to terminal
```

## 10. Limitations and Edge Cases

### 10.1 Unsupported Features

Some Python features cause graph breaks:
- `print()` statements
- Data-dependent control flow: `if x.item() > 0:`
- In-place list/dict modifications
- Some third-party libraries

**Workaround**: Minimize usage in hot path or use `torch._dynamo.allow_in_graph()`

### 10.2 Overhead

Compilation has overhead:
- First call is slow (compilation time)
- Amortized over many calls
- Best for: Repeated calls with same shapes

**Not ideal for**:
- Single-shot inference
- Highly dynamic models
- Rapid prototyping

### 10.3 Memory Usage

Compiled models may use more memory:
- Cached compiled code
- Guard storage
- Workspace memory (CUDA graphs)

## 11. Integration with nn.Module

When you compile a module, it wraps the `forward` method:

```python
model = torch.nn.Linear(128, 10)
compiled_model = torch.compile(model)

# What happens:
# model._call_impl is replaced with compiled version
# Original forward is still accessible
```

**Location**: See `.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:~1771`

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)  # ← Compiled path
    else:
        return self._call_impl(*args, **kwargs)  # ← Normal path
```

## 12. Performance Tips

### 12.1 Choose the Right Mode

```python
# For inference with small batches
model = torch.compile(model, mode="reduce-overhead")

# For maximum performance (longer compilation)
model = torch.compile(model, mode="max-autotune")

# For standard use cases
model = torch.compile(model)  # mode="default"
```

### 12.2 Minimize Graph Breaks

```python
# Bad: print causes graph break
@torch.compile
def f(x):
    y = x + 1
    print(y.shape)  # Graph break!
    return y.relu()

# Good: Remove prints in compiled regions
@torch.compile
def f(x):
    y = x + 1
    return y.relu()

# Or use guards to print outside
def f(x):
    print(f"Input shape: {x.shape}")
    return compiled_f(x)

compiled_f = torch.compile(lambda x: (x + 1).relu())
```

### 12.3 Use Dynamic Shapes Wisely

```python
# If shapes truly vary:
model = torch.compile(model, dynamic=True)

# If shapes are mostly fixed with occasional variation:
model = torch.compile(model)  # dynamic=None (auto)

# If shapes are always fixed:
model = torch.compile(model, dynamic=False)  # Most optimal
```

## 13. Important Source Files

| File | Purpose |
|------|---------|
| `torch/__init__.py:~2478` | `torch.compile()` entry point |
| `torch/_dynamo/eval_frame.py` | `optimize()` function, OptimizeContext |
| `torch/_dynamo/convert_frame.py` | Frame conversion and bytecode analysis |
| `torch/_dynamo/symbolic_convert.py` | Symbolic execution and graph building |
| `torch/_guards.py` | Guards system |
| `torch/_inductor/compile_fx.py` | Inductor backend entry |
| `torch/_inductor/codegen/triton.py` | Triton code generation |
| `torch/fx/graph.py` | FX Graph representation |

## 14. Comparison with Eager Mode

| Aspect | Eager Mode | torch.compile() |
|--------|-----------|-----------------|
| **Execution** | Operation-by-operation | Graph compilation + execution |
| **First run** | Fast | Slow (compilation) |
| **Subsequent runs** | Same speed | Much faster |
| **Flexibility** | Full Python | Some limitations |
| **Memory** | Lower | Higher (caching) |
| **Debugging** | Easy | Harder (compiled code) |
| **Optimization** | None | Fusion, layout, etc. |
| **Best for** | Development, dynamic models | Production, static models |

## 15. Comparison with TorchScript

| Aspect | TorchScript (torch.jit) | torch.compile() |
|--------|------------------------|-----------------|
| **Paradigm** | Ahead-of-time (AOT) | Just-in-time (JIT) |
| **Python support** | Subset only | Most Python features |
| **Graph capture** | Static tracing/scripting | Dynamic bytecode analysis |
| **Backend** | TorchScript interpreter | Inductor/Triton |
| **Performance** | Good | Better |
| **Status** | Legacy | Current recommendation |

## Summary

`torch.compile()` is PyTorch's modern compilation system that:

1. **Entry**: Wraps models/functions with `OptimizeContext`
2. **Capture**: TorchDynamo intercepts Python bytecode, builds FX graphs
3. **Guards**: Generates runtime checks for compiled code validity
4. **Compilation**: Backend (Inductor) optimizes and generates code (Triton/C++)
5. **Caching**: Stores compiled functions with guards
6. **Execution**: Runs optimized code when guards pass
7. **Recompilation**: Handles shape changes, guard failures

**Key Innovation**: No need to rewrite PyTorch code—compile existing models with one decorator.

**Philosophy**:
- `torch.compile()` = **User experience** (compile any PyTorch code)
- **TorchDynamo** = **Graph capture** (bytecode → FX graph)
- **TorchInductor** = **Optimization** (FX graph → fast code)
- **Guards** = **Correctness** (ensure compiled code is valid)

The architecture cleanly separates graph capture from optimization, enabling:
- Support for arbitrary Python code
- Pluggable backends
- Transparent performance improvements without code changes
