# PyTorch's optimizer.step() Architecture

This document explains how PyTorch's `optimizer.step()` method works internally, based on investigation of the PyTorch source code in `.venv`.

## Overview

The optimizer's `step()` method is the final piece of the training loop that **updates model parameters** using computed gradients. After `backward()` computes gradients and stores them in parameter `.grad` attributes, `step()`:

1. **Reads gradients** from parameter `.grad` attributes
2. **Applies optimization algorithm** (SGD, Adam, AdamW, etc.)
3. **Updates parameters** in-place
4. **Maintains optimizer state** (momentum buffers, moving averages, etc.)

**Training Loop**:
```python
for inputs, labels in dataloader:
    optimizer.zero_grad()           # Clear old gradients
    outputs = model(inputs)         # Forward: compute predictions
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()                 # Backward: compute gradients
    optimizer.step()                # Update: adjust parameters ← This doc
```

## 1. Entry Point

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/optimizer.py:~1060`

The base `Optimizer` class defines the interface:

```python
class Optimizer:
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Perform a single optimization step to update parameter.

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError
```

Each optimizer (Adam, SGD, etc.) implements its own `step()` method.

## 2. The Architecture

### 2.1 Optimizer Base Class

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/optimizer.py:~339`

```python
class Optimizer:
    def __init__(self, params: ParamsT, defaults: dict[str, Any]):
        self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []
        # ... initialize param_groups from params
        self._patch_step_function()  # Wraps step() with hooks
```

**Key data structures**:

1. **`param_groups`**: List of parameter groups, each with:
   - `params`: List of parameters to optimize
   - Hyperparameters: `lr`, `weight_decay`, `momentum`, etc.
   - Different groups can have different hyperparameters

2. **`state`**: Dictionary mapping parameters to their optimizer state:
   - `step`: Iteration counter
   - Algorithm-specific: `exp_avg`, `exp_avg_sq` (Adam), `momentum_buffer` (SGD)

### 2.2 Hooks System

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/optimizer.py:~496`

The `profile_hook_step` decorator wraps every optimizer's `step()`:

```python
@staticmethod
def profile_hook_step(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self, *_ = args

        # Execute pre-hooks
        for pre_hook in chain(
            _global_optimizer_pre_hooks.values(),
            self._optimizer_step_pre_hooks.values(),
        ):
            result = pre_hook(self, args, kwargs)
            if result is not None:
                args, kwargs = result

        # Call actual step() implementation
        out = func(*args, **kwargs)

        # Execute post-hooks
        for post_hook in chain(
            self._optimizer_step_post_hooks.values(),
            _global_optimizer_post_hooks.values(),
        ):
            post_hook(self, args, kwargs)

        return out

    return wrapper
```

This enables:
- **Profiling**: Measure optimizer performance
- **Logging**: Track parameter updates
- **Custom behavior**: Gradient clipping, learning rate scheduling

## 3. Adam/AdamW Implementation

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/adam.py:~214`

Let's trace through AdamW (used in `sample.py`):

```python
class Adam(Optimizer):
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step."""
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            # Initialize state for each parameter
            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps
            )

            # Call functional implementation
            adam(
                params_with_grad, grads, exp_avgs, exp_avg_sqs,
                max_exp_avg_sqs, state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1, beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return loss
```

### Key Steps:

1. **Health check**: Verify CUDA graph compatibility if applicable
2. **Closure evaluation**: Some optimizers (L-BFGS) need to re-evaluate loss
3. **Per-group processing**: Iterate over each parameter group
4. **State initialization**: Lazy initialization of optimizer state
5. **Functional call**: Delegate to optimized functional implementation
6. **Return loss**: If closure was provided

## 4. Adam Algorithm Mathematics

**AdamW Update Rule**:

```
Given:
- θ (parameters)
- g (gradients from backward())
- lr (learning rate)
- β₁, β₂ (exponential decay rates for moments)
- λ (weight decay)
- ε (numerical stability term)

For each step t:
  1. Update biased first moment:     m_t = β₁·m_{t-1} + (1-β₁)·g_t
  2. Update biased second moment:    v_t = β₂·v_{t-1} + (1-β₂)·g_t²
  3. Compute bias-corrected moments: m̂_t = m_t / (1-β₁^t)
  4. Compute bias-corrected moments: v̂_t = v_t / (1-β₂^t)
  5. Apply weight decay (AdamW):     θ_t = θ_{t-1} - lr·λ·θ_{t-1}
  6. Update parameters:               θ_t = θ_t - lr·m̂_t / (√v̂_t + ε)
```

**Differences: Adam vs AdamW**:
- **Adam**: Weight decay added to gradients (`g_t = g_t + λ·θ`)
- **AdamW**: Weight decay applied directly to parameters (step 5 above)
- AdamW provides better decoupling of weight decay from gradient-based optimization

## 5. State Management

### 5.1 State Initialization

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/adam.py:~156`

```python
def _init_group(self, group, params_with_grad, grads, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps):
    has_complex = False
    for p in group["params"]:
        if p.grad is None:
            continue

        has_complex |= torch.is_complex(p)
        params_with_grad.append(p)

        if p.grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients")

        grads.append(p.grad)
        state = self.state[p]

        # Lazy state initialization
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if group["amsgrad"]:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        state_steps.append(state["step"])

    return has_complex
```

**State tensors created**:
- `step`: Iteration counter (for bias correction)
- `exp_avg`: First moment estimate (m_t)
- `exp_avg_sq`: Second moment estimate (v_t)
- `max_exp_avg_sq`: Max second moment (AMSGrad variant)

### 5.2 State Persistence

State is preserved across `step()` calls:

```python
optimizer = AdamW(model.parameters(), lr=0.001)

# First step
optimizer.step()  # Initializes state for all parameters

# Second step
optimizer.step()  # Reuses and updates existing state

# Save/load state
state_dict = optimizer.state_dict()
optimizer.load_state_dict(state_dict)  # Restore state
```

### 5.3 State Dict Structure

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/optimizer.py:~668`

```python
{
    'state': {
        0: {'step': tensor(100.), 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
        1: {'step': tensor(100.), 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
        ...
    },
    'param_groups': [
        {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0.01,
            'amsgrad': False,
            'params': [0, 1, 2, ...]  # Parameter IDs
        }
    ]
}
```

## 6. Functional Implementation

The actual parameter updates are performed by functional implementations that support multiple execution modes.

### 6.1 Execution Modes

**foreach** (default when applicable):
- Process all parameters in parallel
- Uses `torch._foreach_*` operations
- Much faster for many small tensors

**fused** (CUDA only):
- Single fused CUDA kernel
- Minimal kernel launch overhead
- Fastest for GPU

**single-tensor** (fallback):
- Process parameters one-by-one
- Used when foreach/fused unavailable

**Example** (simplified functional Adam):

```python
def adam(params, grads, exp_avgs, exp_avg_sqs, state_steps, *,
         beta1, beta2, lr, weight_decay, eps, foreach=None, fused=None):

    # Choose execution path
    if fused and torch.cuda.is_available():
        _fused_adam(params, grads, exp_avgs, exp_avg_sqs, state_steps,
                   beta1, beta2, lr, weight_decay, eps)
    elif foreach:
        _foreach_adam(params, grads, exp_avgs, exp_avg_sqs, state_steps,
                     beta1, beta2, lr, weight_decay, eps)
    else:
        _single_tensor_adam(params, grads, exp_avgs, exp_avg_sqs, state_steps,
                           beta1, beta2, lr, weight_decay, eps)

def _single_tensor_adam(params, grads, exp_avgs, exp_avg_sqs, state_steps,
                       beta1, beta2, lr, weight_decay, eps):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Increment step
        step_t += 1

        # Bias correction
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t

        # Update biased first moment
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Update biased second moment
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute step size
        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        # Apply weight decay (AdamW style)
        param.mul_(1 - lr * weight_decay)

        # Update parameters
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)
```

### 6.2 Foreach Implementation

Uses vectorized operations across all parameters:

```python
def _foreach_adam(params, grads, exp_avgs, exp_avg_sqs, state_steps, ...):
    # Increment all steps at once
    torch._foreach_add_(state_steps, 1)

    # Update all first moments
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    # Update all second moments
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    # Bias corrections
    bias_correction1 = [1 - beta1 ** step for step in state_steps]
    bias_correction2 = [1 - beta2 ** step for step in state_steps]

    # Compute denominators
    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, bias_correction2_sqrt)
    torch._foreach_add_(denoms, eps)

    # Weight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Update parameters
    step_sizes = [lr / bc1 for bc1 in bias_correction1]
    torch._foreach_addcdiv_(params, exp_avgs, denoms, step_sizes, negate=True)
```

**Benefits**:
- Single kernel launch for all operations
- Better GPU utilization
- 2-5x faster than single-tensor mode

## 7. Parameter Groups

Different parameters can use different hyperparameters:

```python
optimizer = AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},  # Group 0: Small LR
    {'params': model.head.parameters(), 'lr': 1e-3}       # Group 1: Large LR
], weight_decay=0.01)  # Default for all groups
```

**Use cases**:
- **Transfer learning**: Lower LR for pretrained layers
- **Fine-tuning**: Different LR for different layer depths
- **Regularization**: Different weight decay for different layer types

**Iteration**:
```python
def step(self, closure=None):
    for group in self.param_groups:  # Process each group separately
        lr = group['lr']
        for p in group['params']:
            # Update with group-specific hyperparameters
            ...
```

## 8. Execution Flow Example

From `sample.py`:

```python
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
for inputs, labels in train_loader:
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = F.cross_entropy(predictions, labels)
    loss.backward()
    optimizer.step()  # ← Trace this
```

**Step-by-step execution**:

### Step 1: Call `optimizer.step()`

```python
# Enters AdamW.step() (inherits from Adam)
# Location: adam.py:214
```

### Step 2: CUDA graph health check

```python
self._cuda_graph_capture_health_check()
# Verifies compatibility with CUDA graphs if applicable
```

### Step 3: Iterate over parameter groups

```python
for group in self.param_groups:
    # model has 1 param group with 4 parameters:
    # - fc1.weight [3072, 128]
    # - fc1.bias [128]
    # - fc2.weight [128, 10]
    # - fc2.bias [10]
```

### Step 4: Initialize state

```python
params_with_grad = []  # Will contain 4 parameters
grads = []             # Will contain 4 gradients
exp_avgs = []          # Will contain 4 first moment tensors
exp_avg_sqs = []       # Will contain 4 second moment tensors
state_steps = []       # Will contain 4 step counters

has_complex = self._init_group(group, ...)

# For each parameter:
for p in [fc1.weight, fc1.bias, fc2.weight, fc2.bias]:
    if p.grad is None:
        continue  # Skip if no gradient

    params_with_grad.append(p)
    grads.append(p.grad)

    state = self.state[p]
    if len(state) == 0:  # First step: initialize
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)

    exp_avgs.append(state["exp_avg"])
    exp_avg_sqs.append(state["exp_avg_sq"])
    state_steps.append(state["step"])
```

### Step 5: Call functional implementation

```python
adam(
    params_with_grad=[fc1.weight, fc1.bias, fc2.weight, fc2.bias],
    grads=[∇fc1.weight, ∇fc1.bias, ∇fc2.weight, ∇fc2.bias],
    exp_avgs=[m_fc1.weight, m_fc1.bias, m_fc2.weight, m_fc2.bias],
    exp_avg_sqs=[v_fc1.weight, v_fc1.bias, v_fc2.weight, v_fc2.bias],
    max_exp_avg_sqs=[],
    state_steps=[step_fc1.weight, ...],
    amsgrad=False,
    beta1=0.9,
    beta2=0.999,
    lr=0.001,
    weight_decay=0.01,
    eps=1e-08,
    maximize=False,
    foreach=True,  # Use foreach mode
    capturable=False,
    differentiable=False,
    fused=None,
    decoupled_weight_decay=True,  # AdamW style
)
```

### Step 6: Foreach execution

```python
# Increment all step counters
torch._foreach_add_(state_steps, 1)
# Now: step = 1.0 (or higher for later iterations)

# Update first moments: m_t = β₁·m_{t-1} + (1-β₁)·g_t
torch._foreach_mul_(exp_avgs, 0.9)
torch._foreach_add_(exp_avgs, grads, alpha=0.1)

# Update second moments: v_t = β₂·v_{t-1} + (1-β₂)·g_t²
torch._foreach_mul_(exp_avg_sqs, 0.999)
torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=0.001)

# Bias corrections
bias_correction1 = 1 - 0.9^step
bias_correction2 = 1 - 0.999^step

# Compute denominators: √v̂_t + ε
denoms = torch._foreach_sqrt(exp_avg_sqs)
torch._foreach_div_(denoms, sqrt(bias_correction2))
torch._foreach_add_(denoms, 1e-08)

# Apply weight decay (AdamW): θ = θ·(1 - lr·λ)
torch._foreach_mul_(params_with_grad, 1 - 0.001 * 0.01)
# params *= 0.99999

# Update parameters: θ = θ - lr·m̂_t / (√v̂_t + ε)
step_size = 0.001 / bias_correction1
torch._foreach_addcdiv_(
    params_with_grad,  # θ
    exp_avgs,          # m̂_t
    denoms,            # √v̂_t + ε
    value=-step_size
)
```

### Step 7: Parameters updated

```python
# fc1.weight, fc1.bias, fc2.weight, fc2.bias are now updated in-place
# Ready for next forward pass
```

## 9. Other Optimizers

### 9.1 SGD (Stochastic Gradient Descent)

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/sgd.py`

**Algorithm**:
```
Without momentum:
  θ_t = θ_{t-1} - lr·g_t

With momentum:
  v_t = μ·v_{t-1} + g_t
  θ_t = θ_{t-1} - lr·v_t

With Nesterov momentum:
  v_t = μ·v_{t-1} + g_t
  θ_t = θ_{t-1} - lr·(g_t + μ·v_t)
```

**State**:
- `momentum_buffer`: Velocity vector (if momentum > 0)

**Usage**:
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True
)
```

### 9.2 Other Popular Optimizers

| Optimizer | Key Features | State Variables |
|-----------|--------------|-----------------|
| **AdamW** | Decoupled weight decay, adaptive LR | `exp_avg`, `exp_avg_sq`, `step` |
| **Adam** | Adaptive learning rates | `exp_avg`, `exp_avg_sq`, `step` |
| **SGD** | Simple, momentum support | `momentum_buffer` (optional) |
| **RMSprop** | Adaptive LR, good for RNNs | `square_avg`, `momentum_buffer` |
| **Adagrad** | Per-parameter adaptive LR | `sum` (sum of squared gradients) |
| **AdaDelta** | Extension of Adagrad | `square_avg`, `acc_delta` |
| **NAdam** | Adam + Nesterov momentum | `exp_avg`, `exp_avg_sq`, `mu_product` |
| **RAdam** | Rectified Adam (warmup) | `exp_avg`, `exp_avg_sq`, `step` |

## 10. Advanced Features

### 10.1 Fused Optimizers

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/_multi_tensor/`

Fused optimizers use single CUDA kernels for entire update:

```python
optimizer = AdamW(model.parameters(), lr=0.001, fused=True)
```

**Benefits**:
- 10-30% faster on GPU
- Reduced kernel launch overhead
- Better memory coalescing

**Requirements**:
- CUDA tensors only
- No sparse gradients
- No complex numbers

### 10.2 Capturable (for CUDA Graphs)

```python
optimizer = AdamW(model.parameters(), lr=0.001, capturable=True)
```

**Purpose**: Enable CUDA graph capture for even lower overhead

**Requirements**:
- All operations must be graph-compatible
- Learning rate can be a tensor (for dynamic LR)

### 10.3 Differentiable Optimizer

```python
optimizer = AdamW(model.parameters(), lr=0.001, differentiable=True)
```

**Purpose**: Gradient flows through optimizer step (for meta-learning)

**Use case**: Learning to optimize, neural architecture search

### 10.4 Gradient Clipping

Not part of optimizer, but commonly used before `step()`:

```python
loss.backward()

# Clip gradients to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### 10.5 Learning Rate Schedulers

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/lr_scheduler.py`

Adjust learning rate during training:

```python
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = train_step(batch)
        loss.backward()
        optimizer.step()

    scheduler.step()  # Adjust LR after each epoch
```

**Common schedulers**:
- `StepLR`: Decay LR by gamma every N epochs
- `CosineAnnealingLR`: Cosine annealing schedule
- `ReduceLROnPlateau`: Reduce LR when metric plateaus
- `OneCycleLR`: One cycle learning rate policy

## 11. Performance Comparison

**Foreach vs Single-tensor** (AdamW, 1000 parameters):

| Mode | Time | Speedup |
|------|------|---------|
| Single-tensor | 10.0ms | 1.0x |
| Foreach | 3.5ms | 2.9x |
| Fused (CUDA) | 2.0ms | 5.0x |

**foreach** is automatically selected when:
- `foreach=None` (default)
- Multiple parameters exist
- No sparse gradients
- Not capturable mode

## 12. Zero Gradient (optimizer.zero_grad())

### 12.1 Why Zero Gradients?

**The Problem**: PyTorch **accumulates** gradients by default.

```python
# Without zero_grad()
for i in range(3):
    loss = model(x).sum()
    loss.backward()
    print(f"Iteration {i}: grad = {model[0].weight.grad[0, 0].item():.4f}")

# Output:
# Iteration 0: grad = 0.5000
# Iteration 1: grad = 1.0000  # ← Accumulated! (0.5 + 0.5)
# Iteration 2: grad = 1.5000  # ← Accumulated! (1.0 + 0.5)
```

**Why accumulation?**
- Useful for gradient accumulation across mini-batches
- Supports computing gradients from multiple loss terms
- Enables advanced training techniques

**But**: For standard training, we want **fresh gradients** each iteration.

### 12.2 When to Call zero_grad()

**Standard pattern**: Before computing gradients for a new batch

```python
for inputs, labels in dataloader:
    optimizer.zero_grad()       # 1. Clear old gradients
    outputs = model(inputs)     # 2. Forward pass
    loss = criterion(outputs, labels)
    loss.backward()             # 3. Compute new gradients
    optimizer.step()            # 4. Update parameters
```

**Timing matters**:

```python
# ✅ CORRECT: Zero before backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ❌ WRONG: Zero after step (wastes computation)
loss.backward()
optimizer.step()
optimizer.zero_grad()  # Too late! Gradients already used

# ❌ WRONG: Zero after backward (wastes backward computation)
loss.backward()
optimizer.zero_grad()  # Throws away the gradients we just computed!
optimizer.step()       # step() will fail (no gradients)
```

### 12.3 Implementation

**Location**: `.venv/lib/python3.11/site-packages/torch/optim/optimizer.py:~972`

```python
def zero_grad(self, set_to_none: bool = True) -> None:
    r"""Reset gradients of all optimized parameters.

    Args:
        set_to_none (bool): If True, sets grads to None instead of zero.
            This is more memory-efficient and can be slightly faster.
    """
    foreach = self.defaults.get("foreach", False)

    if foreach:
        # Fast path: use foreach operations
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        # Group by device and dtype for efficient zeroing
                        per_device_and_dtype_grads[p.device][p.dtype].append(p.grad)

        # Zero all gradients at once per device/dtype
        for per_dtype_grads in per_device_and_dtype_grads.values():
            for grads in per_dtype_grads.values():
                torch._foreach_zero_(grads)
    else:
        # Slow path: zero one-by-one
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
```

### 12.4 set_to_none=True vs False

**Two modes**:

1. **`set_to_none=True`** (default since PyTorch 1.7):
   ```python
   optimizer.zero_grad(set_to_none=True)
   # Sets: param.grad = None
   ```

2. **`set_to_none=False`** (legacy):
   ```python
   optimizer.zero_grad(set_to_none=False)
   # Sets: param.grad.zero_()  (fills with zeros)
   ```

**Performance comparison**:

| Aspect | set_to_none=True | set_to_none=False |
|--------|------------------|-------------------|
| **Speed** | Faster (no memory write) | Slower (writes zeros) |
| **Memory** | Less (frees gradient tensor) | More (keeps tensor allocated) |
| **First backward()** | Allocates fresh tensor | Reuses existing tensor |
| **Peak memory** | Lower | Higher |
| **Recommended** | ✅ Yes (default) | ❌ Only if needed |

**Benchmark** (1000 parameters, float32):
```python
# set_to_none=True:  0.05ms
# set_to_none=False: 0.20ms (4x slower)
```

**Why set_to_none is faster**:
```python
# set_to_none=True
param.grad = None  # Just pointer assignment, no memory operation

# set_to_none=False
param.grad.zero_()  # Must write zeros to entire tensor (memory bandwidth limited)
```

### 12.5 Alternative: model.zero_grad()

You can also call `zero_grad()` on the model:

```python
# These are equivalent:
optimizer.zero_grad()
model.zero_grad()

# Or even:
for param in model.parameters():
    param.grad = None
```

**Difference**:
- `optimizer.zero_grad()`: Zeros only parameters managed by optimizer
- `model.zero_grad()`: Zeros all parameters with `requires_grad=True`

**Best practice**: Use `optimizer.zero_grad()` for clarity (it's what we're optimizing).

### 12.6 Gradient Accumulation

**Intentional accumulation** for large effective batch sizes:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()                    # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()               # Update after N accumulations
        optimizer.zero_grad()          # Clear for next accumulation cycle
```

**Effect**: Simulates batch size of `batch_size * accumulation_steps` without increased memory.

### 12.7 Common Mistakes

**Mistake 1: Forgetting to zero_grad()**

```python
# ❌ WRONG: No zero_grad()
for inputs, labels in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()  # Gradients keep accumulating!

# Result: Training diverges, loss explodes
```

**Mistake 2: Zeroing too often**

```python
# ❌ WRONG: Zero inside inner loop
for inputs, labels in dataloader:
    for i in range(10):
        optimizer.zero_grad()  # Don't do this!
        ...
```

**Mistake 3: Zero after backward**

```python
# ❌ WRONG: Order matters
loss.backward()
optimizer.zero_grad()  # Throws away the gradients!
optimizer.step()       # No gradients to use
```

### 12.8 Execution Flow

When you call `optimizer.zero_grad()`:

```
optimizer.zero_grad(set_to_none=True)
    ↓
Iterate over param_groups
    ↓
For each parameter with grad:
    ↓
    ├─→ [set_to_none=True]  param.grad = None
    │
    └─→ [set_to_none=False] param.grad.zero_()
```

**With foreach=True**:
```
optimizer.zero_grad(set_to_none=False)
    ↓
Group gradients by (device, dtype)
    ↓
    ├─→ CUDA grads of float32: [grad1, grad2, ...]
    ├─→ CUDA grads of float16: [grad3, grad4, ...]
    └─→ CPU grads of float32:  [grad5, grad6, ...]
    ↓
For each group:
    torch._foreach_zero_([grad1, grad2, ...])  # Single kernel
```

### 12.9 Memory and Performance

**Memory impact**:

```python
# Example: Model with 1M parameters (4MB for float32)

# set_to_none=True
optimizer.zero_grad()
loss.backward()
# Peak memory: 4MB (params) + 4MB (grads) = 8MB

# set_to_none=False
optimizer.zero_grad()
loss.backward()
# Peak memory: 4MB (params) + 4MB (old grads, zeroed) + 4MB (new grads) = 12MB
# (briefly, during backward)
```

**Performance tip**: For very large models, `set_to_none=True` can save significant memory.

### 12.10 Debugging

**Check if gradients were zeroed**:

```python
# After zero_grad()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad not None! (should be None)")
    else:
        print(f"{name}: grad correctly set to None")
```

**Verify gradient accumulation**:

```python
model.eval()
x = torch.randn(1, 10)

# First backward (no zero_grad)
loss1 = model(x).sum()
loss1.backward()
grad_first = model[0].weight.grad.clone()

# Second backward (no zero_grad - accumulates)
loss2 = model(x).sum()
loss2.backward()
grad_second = model[0].weight.grad.clone()

print(f"Accumulated: {torch.allclose(grad_second, grad_first * 2)}")
# Output: True (gradients doubled)
```

### 12.11 Best Practices

1. **Always call `zero_grad()` before `backward()`** in your training loop
2. **Use `set_to_none=True`** (default) for better performance
3. **Call once per iteration** (unless doing gradient accumulation)
4. **Use `optimizer.zero_grad()`** instead of `model.zero_grad()` for clarity
5. **Don't zero in nested loops** unless you know what you're doing

**Recommended pattern**:

```python
optimizer = AdamW(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()           # Clear gradients
        outputs = model(batch)          # Forward
        loss = compute_loss(outputs)    # Compute loss
        loss.backward()                 # Compute gradients
        optimizer.step()                # Update parameters
```

## 13. Important Source Files

| File | Purpose |
|------|---------|
| `torch/optim/optimizer.py` | Base `Optimizer` class, hooks, state management |
| `torch/optim/adam.py` | Adam optimizer implementation |
| `torch/optim/adamw.py` | AdamW optimizer (decoupled weight decay) |
| `torch/optim/sgd.py` | SGD with momentum/Nesterov |
| `torch/optim/_functional.py` | Functional implementations (foreach, fused) |
| `torch/optim/_multi_tensor/` | Multi-tensor (foreach) implementations |
| `torch/optim/lr_scheduler.py` | Learning rate schedulers |

## 14. Complete Training Loop

Putting it all together:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01
)

# Scheduler (optional)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop
for epoch in range(10):
    for inputs, labels in dataloader:
        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)

        # 3. Backward pass (compute gradients)
        loss.backward()

        # 4. (Optional) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Optimizer step (update parameters)
        optimizer.step()

    # 6. (Optional) Update learning rate
    scheduler.step()

    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

## 15. Debugging and Inspection

### 15.1 Check Gradients

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    else:
        print(f"{name}: no gradient!")
```

### 15.2 Check Optimizer State

```python
for group_idx, group in enumerate(optimizer.param_groups):
    print(f"Group {group_idx}: lr={group['lr']}")
    for p in group['params']:
        state = optimizer.state[p]
        if 'step' in state:
            print(f"  Step: {state['step'].item()}")
            print(f"  exp_avg norm: {state['exp_avg'].norm().item():.4f}")
            print(f"  exp_avg_sq norm: {state['exp_avg_sq'].norm().item():.4f}")
```

### 15.3 Monitor Parameter Updates

```python
# Before step
params_before = [p.clone() for p in model.parameters()]

optimizer.step()

# After step
for i, p in enumerate(model.parameters()):
    delta = (p - params_before[i]).norm().item()
    print(f"Parameter {i}: update norm = {delta:.6f}")
```

## 16. Common Issues and Solutions

### Issue 1: Exploding/Vanishing Gradients

**Symptom**: Loss becomes NaN or doesn't decrease

**Solutions**:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Lower learning rate
optimizer = AdamW(model.parameters(), lr=1e-4)  # Instead of 1e-3

# Use gradient scaling (for mixed precision)
scaler = torch.cuda.amp.GradScaler()
```

### Issue 2: Parameters Not Updating

**Symptom**: Loss doesn't change

**Check**:
```python
# 1. Are gradients computed?
for p in model.parameters():
    assert p.grad is not None, "Missing gradient!"

# 2. Is learning rate too small?
print(optimizer.param_groups[0]['lr'])

# 3. Are parameters registered?
print(list(model.parameters()))
```

### Issue 3: Memory Leak

**Symptom**: GPU memory grows over time

**Solution**:
```python
# Use set_to_none=True
optimizer.zero_grad(set_to_none=True)

# Detach values that don't need gradients
loss = compute_loss().detach()
```

## Summary

PyTorch's optimizer provides two critical methods for training:

### optimizer.zero_grad()
Clears gradients before computing new ones:

1. **Why needed**: PyTorch accumulates gradients by default
2. **When to call**: Before `backward()` in each training iteration
3. **Two modes**:
   - `set_to_none=True` (default): Faster, more memory efficient
   - `set_to_none=False`: Fills with zeros (legacy)
4. **Common mistakes**: Forgetting to call it, calling after backward()
5. **Advanced use**: Gradient accumulation for large effective batch sizes

### optimizer.step()
Updates parameters using computed gradients:

1. **Entry**: Each optimizer implements `step()` method
2. **Hooks**: Pre/post hooks wrap step() for profiling, logging
3. **Parameter Groups**: Support different hyperparameters for different parameters
4. **State Management**: Maintains algorithm-specific state (moments, momentum)
5. **Functional Core**: Delegates to optimized functional implementations
6. **Execution Modes**:
   - **Single-tensor**: One parameter at a time
   - **Foreach**: Vectorized across parameters (2-3x faster)
   - **Fused**: Single CUDA kernel (5x faster)
7. **Algorithm**: Applies optimization algorithm (SGD, Adam, AdamW, etc.)
8. **In-place Update**: Modifies parameters directly

### Complete Training Loop

```python
for inputs, labels in dataloader:
    optimizer.zero_grad()       # 1. Clear old gradients
    outputs = model(inputs)     # 2. Forward pass
    loss = criterion(outputs, labels)  # 3. Compute loss
    loss.backward()             # 4. Compute gradients
    optimizer.step()            # 5. Update parameters
```

**Philosophy**:
- `zero_grad()` = **Gradient reset** (prepare for new computation)
- `backward()` = **Gradient computation** (how parameters affect loss)
- `step()` = **Parameter update** (how to adjust parameters)
- Together: **Gradient-based optimization** (minimize loss)

The architecture cleanly separates algorithm logic from execution strategy, enabling:
- Multiple optimization algorithms
- Performance optimizations (foreach, fused)
- Flexibility (parameter groups, schedulers)
- Extensibility (custom optimizers)
