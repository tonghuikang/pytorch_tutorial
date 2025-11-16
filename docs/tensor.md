# PyTorch Tensor Storage Architecture

This document explains how PyTorch stores tensor data internally, based on investigation of the PyTorch source code in `.venv`.

## Overview

PyTorch tensors consist of two main components:

1. **Tensor metadata**: Shape, strides, dtype, device
2. **Storage**: The actual raw data in memory

Multiple tensors can share the same underlying storage while presenting different views of the data. This enables efficient operations like slicing, transposing, and reshaping without copying data.

**Key Insight**: A tensor is a **view** into a storage with associated metadata describing how to interpret that storage as a multidimensional array.

## 1. Storage: The Raw Data Container

### 1.1 What is Storage?

**Location**: `.venv/lib/python3.11/site-packages/torch/storage.py`

Storage is a 1-dimensional, contiguous array of bytes in memory. It's the actual data buffer that holds tensor values.

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(list(t.untyped_storage()))
# Storage is always 1D: [1, 2, 3, 4, 5, 6]
```

**Storage vs Tensor**:
- **Storage**: Flat, 1D array of bytes
- **Tensor**: Multidimensional view with shape and strides

### 1.2 Storage Types

PyTorch uses **UntypedStorage** as the modern storage class:

```python
t = torch.randn(2, 3)
storage = t.untyped_storage()

print(type(storage))
# <class 'torch.UntypedStorage'>

print(storage.nbytes())
# 24 (6 elements * 4 bytes per float32)

print(t.data_ptr())
# Memory address, e.g., 4898926592
```

**Important**: `TypedStorage` is deprecated. Always use `untyped_storage()` instead of `storage()`.

### 1.3 Storage Properties

| Property | Description | Example |
|----------|-------------|---------|
| `nbytes()` | Total bytes in storage | `storage.nbytes()` → 24 |
| `data_ptr()` | Memory address | `tensor.data_ptr()` → 4898926592 |
| `device` | Where stored (CPU/CUDA) | `storage.device` → cpu |
| `size()` | Number of bytes | Same as `nbytes()` |

## 2. Tensor Metadata: How to Interpret Storage

**Location**: `.venv/lib/python3.11/site-packages/torch/_tensor.py:~108` (Tensor class), `.venv/lib/python3.11/site-packages/torch/_C/__init__.pyi` (TensorBase properties)

A tensor knows how to interpret its storage through metadata:

```python
t = torch.arange(12).reshape(3, 4)
print('Shape:', t.shape)           # torch.Size([3, 4])
print('Stride:', t.stride())       # (4, 1)
print('Storage offset:', t.storage_offset())  # 0
print('Element size:', t.element_size())      # 8 bytes (int64)
print('Dtype:', t.dtype)           # torch.int64
```

### 2.1 Shape

The dimensions of the tensor:

```python
t = torch.zeros(2, 3, 4)
print(t.shape)  # torch.Size([2, 3, 4])
print(t.ndim)   # 3
```

### 2.2 Strides

**Strides** define how many elements to skip in storage to move one position along each dimension.

```python
t = torch.arange(12).reshape(3, 4)
# Storage: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Shape: [3, 4]

print(t.stride())  # (4, 1)
```

**How strides work**:
- Stride (4, 1) means:
  - Move 4 elements in storage to go down 1 row
  - Move 1 element in storage to go right 1 column

```
To access element t[i, j]:
storage_index = offset + i * stride[0] + j * stride[1]
              = 0 + i * 4 + j * 1

t[1, 2] → storage[0 + 1*4 + 2*1] → storage[6] = 6 ✓
```

### 2.3 Storage Offset

The starting position in storage for this tensor:

```python
t = torch.arange(12).reshape(3, 4)
slice_t = t[1:, ::2]  # Start from row 1, take every 2nd column

print('Original offset:', t.storage_offset())        # 0
print('Slice offset:', slice_t.storage_offset())     # 4
print('Slice:', slice_t)
# tensor([[ 4,  6],
#         [ 8, 10]])
```

The slice starts at index 4 in storage (first element of row 1).

## 3. Storage Sharing: Views vs Copies

### 3.1 Views Share Storage

Operations that create **views** share the underlying storage:

```python
t = torch.arange(6).reshape(2, 3)
print('Original:', t)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

# Transpose is a view
t_T = t.t()
print('Transposed:', t_T)
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])

# Check if they share storage
print('Same storage?', t.data_ptr() == t_T.data_ptr())
# True - same storage!

# Modifying the view affects the original
t_T[0, 0] = 999
print('Original after modification:', t)
# tensor([[999,   1,   2],
#         [  3,   4,   5]])
```

**Common view operations**:
- `transpose()`, `t()`, `T`
- `permute()`
- `view()`, `reshape()` (sometimes)
- Slicing: `t[1:, ::2]`
- `expand()`
- `narrow()`

### 3.2 How Transpose Works (View Magic)

Transpose doesn't move data - it just changes strides:

```python
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
# Storage: [1, 2, 3, 4, 5, 6]
# Shape: (2, 3)
# Stride: (3, 1)

t_T = t.t()
# Storage: [1, 2, 3, 4, 5, 6]  ← Same storage!
# Shape: (3, 2)
# Stride: (1, 3)  ← Just swapped strides!
```

**Original** (stride 3, 1):
```
t[0,0]=storage[0*3+0*1]=storage[0]=1  t[0,1]=storage[1]=2  t[0,2]=storage[2]=3
t[1,0]=storage[1*3+0*1]=storage[3]=4  t[1,1]=storage[4]=5  t[1,2]=storage[5]=6
```

**Transposed** (stride 1, 3):
```
t_T[0,0]=storage[0*1+0*3]=storage[0]=1  t_T[0,1]=storage[3]=4
t_T[1,0]=storage[1*1+0*3]=storage[1]=2  t_T[1,1]=storage[4]=5
t_T[2,0]=storage[2*1+0*3]=storage[2]=3  t_T[2,1]=storage[5]=6
```

### 3.3 When Copies Are Made

Some operations **must** copy data:

```python
t = torch.arange(6).reshape(2, 3)

# clone() always copies
t_copy = t.clone()
print('Same storage?', t.data_ptr() == t_copy.data_ptr())
# False - different storage

# contiguous() copies if needed
t_noncontig = t.t()  # Non-contiguous view
t_contig = t_noncontig.contiguous()  # Copies to make contiguous
print('Same storage?', t_noncontig.data_ptr() == t_contig.data_ptr())
# False - copied
```

**Operations that copy**:
- `clone()`
- `contiguous()` (if not already contiguous)
- `copy_()`
- Moving to different device: `t.cuda()`
- Changing dtype: `t.float()`

## 4. Contiguous vs Non-Contiguous Tensors

### 4.1 What is Contiguous?

**Location**: `.venv/lib/python3.11/site-packages/torch/_C/__init__.pyi:~5840` (`is_contiguous()` method)

A tensor is **contiguous** if its elements are laid out sequentially in memory in the order you'd expect from its shape.

**Mathematical definition**: A tensor is contiguous if:
```
stride[i] == stride[i+1] * size[i+1]  for all i
```

```python
t = torch.arange(12).reshape(3, 4)
print('Contiguous?', t.is_contiguous())  # True
print('Stride:', t.stride())             # (4, 1)
# Check: stride[0] == stride[1] * size[1]
#        4 == 1 * 4  ✓
```

### 4.2 Non-Contiguous Example

```python
t = torch.arange(12).reshape(3, 4)
# Storage: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

t_T = t.t()  # Transpose
print('Transposed contiguous?', t_T.is_contiguous())  # False
print('Transposed stride:', t_T.stride())             # (1, 4)

# Elements are NOT sequential in storage:
# t_T[0] = [0, 4, 8]  → storage indices [0, 4, 8] (not sequential!)
```

### 4.3 Non-Contiguous from Slicing

```python
t = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Take every other column
t_slice = t[:, ::2]
# tensor([[ 0,  2],
#         [ 4,  6],
#         [ 8, 10]])

print('Sliced contiguous?', t_slice.is_contiguous())  # False
print('Sliced stride:', t_slice.stride())             # (4, 2)
print('Storage:', list(t_slice.untyped_storage()))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] - full storage!
```

The slice has stride (4, 2), skipping elements:
- Row 0: indices 0, 2 (skip 1, 3)
- Row 1: indices 4, 6 (skip 5, 7)
- Row 2: indices 8, 10 (skip 9, 11)

### 4.4 Making Tensors Contiguous

Use `contiguous()` to create a contiguous copy:

```python
t_noncontig = torch.arange(12).reshape(3, 4).t()
print('Before:', t_noncontig.is_contiguous())  # False

t_contig = t_noncontig.contiguous()
print('After:', t_contig.is_contiguous())      # True

# Now has new storage with sequential layout
print('Same storage?', t_noncontig.data_ptr() == t_contig.data_ptr())
# False - contiguous() made a copy
```

### 4.5 Why Contiguity Matters

Many operations require contiguous tensors for performance:

```python
t = torch.arange(12).reshape(3, 4).t()  # Non-contiguous

# view() requires contiguous
try:
    t.view(-1)  # Flatten
except RuntimeError as e:
    print(e)
    # view size is not compatible with input tensor's size and stride

# Solution: make contiguous first
t_flat = t.contiguous().view(-1)
print(t_flat)  # Works!
```

**Operations requiring contiguity**:
- `view()` (reshape sometimes works without)
- Many CUDA operations
- Some neural network layers

## 5. Data Types and Element Size

### 5.1 Common Data Types

```python
t_float32 = torch.randn(10)
t_float64 = torch.randn(10, dtype=torch.float64)
t_int32 = torch.randint(0, 100, (10,), dtype=torch.int32)

print('float32 element size:', t_float32.element_size())  # 4 bytes
print('float64 element size:', t_float64.element_size())  # 8 bytes
print('int32 element size:', t_int32.element_size())      # 4 bytes
```

**Common dtypes**:

| dtype | Bytes | Description |
|-------|-------|-------------|
| `torch.float32` (`float`) | 4 | Default floating point |
| `torch.float64` (`double`) | 8 | Double precision |
| `torch.float16` (`half`) | 2 | Half precision (GPU) |
| `torch.bfloat16` | 2 | Brain float (ML training) |
| `torch.int64` (`long`) | 8 | Default integer |
| `torch.int32` (`int`) | 4 | 32-bit integer |
| `torch.bool` | 1 | Boolean |

### 5.2 Storage Size Calculation

```python
t = torch.randn(3, 4, 5)  # 60 elements

bytes_per_element = t.element_size()  # 4 (float32)
total_bytes = t.numel() * bytes_per_element
storage_bytes = t.untyped_storage().nbytes()

print(f'Elements: {t.numel()}')              # 60
print(f'Bytes per element: {bytes_per_element}')  # 4
print(f'Total bytes: {total_bytes}')         # 240
print(f'Storage bytes: {storage_bytes}')     # 240
```

## 6. Device Placement

### 6.1 CPU vs CUDA Storage

```python
# CPU tensor
t_cpu = torch.randn(10)
print('Device:', t_cpu.device)  # cpu
print('Storage device:', t_cpu.untyped_storage().device)  # cpu

# CUDA tensor (if available)
if torch.cuda.is_available():
    t_cuda = t_cpu.cuda()
    print('Device:', t_cuda.device)  # cuda:0
    print('Storage device:', t_cuda.untyped_storage().device)  # cuda:0

    # Different storage!
    print('Same storage?', t_cpu.data_ptr() == t_cuda.data_ptr())
    # False - data copied to GPU
```

### 6.2 Moving Between Devices

```python
t_cpu = torch.randn(10)

# Move to CUDA
t_cuda = t_cpu.to('cuda')  # Copies data to GPU

# Move back to CPU
t_cpu_again = t_cuda.to('cpu')  # Copies data back

# No-op if already on device
t_cpu2 = t_cpu.to('cpu')
print('Same storage?', t_cpu.data_ptr() == t_cpu2.data_ptr())
# True - no copy needed
```

## 7. Memory-Mapped Storage

### 7.1 Loading from File

PyTorch can create storage backed by memory-mapped files:

```python
# Create a memory-mapped storage
filename = '/tmp/tensor_data.bin'
storage = torch.UntypedStorage.from_file(
    filename,
    shared=True,  # MAP_SHARED for process sharing
    nbytes=1024   # 1KB
)
```

**Use cases**:
- Loading huge datasets that don't fit in RAM
- Sharing tensors between processes
- Fast loading without copying to RAM

## 8. Storage Implementation Details

### 8.1 Base Class

**Location**: `.venv/lib/python3.11/site-packages/torch/storage.py:~41`

```python
class _StorageBase:
    _cdata: Any  # C++ storage object
    device: torch.device

    def nbytes(self) -> int:
        """Return number of bytes in storage"""

    def data_ptr(self) -> int:
        """Return memory address of storage"""
```

### 8.2 UntypedStorage

**Modern storage class** (replaces TypedStorage):

```python
t = torch.randn(10)
storage = t.untyped_storage()

# Storage operations
print(storage.nbytes())      # Total bytes
print(storage.size())        # Same as nbytes()
print(storage.device)        # cpu or cuda
storage_copy = storage.clone()  # Deep copy
```

### 8.3 C++ Implementation

The actual storage is implemented in C++:
- **CPU storage**: Uses standard memory allocation (malloc/new)
- **CUDA storage**: Uses CUDA memory APIs (cudaMalloc)
- **Memory pooling**: PyTorch caches allocations for reuse

## 9. Practical Examples

### 9.1 Efficient Tensor Slicing

```python
# Large tensor
big_tensor = torch.randn(1000, 1000)

# Slicing creates a view (no copy!)
slice_view = big_tensor[:100, :100]

print('Same storage?', big_tensor.data_ptr() == slice_view.data_ptr())
# True

# Modifying the slice affects original
slice_view.fill_(0)
print(big_tensor[:5, :5])  # First 5x5 is now zeros!
```

### 9.2 Avoiding Unnecessary Copies

```python
t = torch.randn(100, 100)

# ❌ BAD: Creates unnecessary copy
t_bad = t.clone().transpose(0, 1).clone()

# ✅ GOOD: Just a view
t_good = t.t()

# Only copy when necessary
t_contig = t.t().contiguous() if not t.t().is_contiguous() else t.t()
```

### 9.3 In-Place Operations on Shared Storage

```python
t1 = torch.arange(6).reshape(2, 3)
t2 = t1  # Shares storage!

# In-place operation affects both
t1.add_(10)

print(t1)
# tensor([[10, 11, 12],
#         [13, 14, 15]])

print(t2)  # Also changed!
# tensor([[10, 11, 12],
#         [13, 14, 15]])

# To avoid this, clone first
t3 = t1.clone()
t3.add_(100)
print(t1)  # Unchanged
```

### 9.4 Debugging Storage Issues

```python
def print_tensor_info(t, name="Tensor"):
    print(f"\n{name}:")
    print(f"  Shape: {t.shape}")
    print(f"  Stride: {t.stride()}")
    print(f"  Offset: {t.storage_offset()}")
    print(f"  Contiguous: {t.is_contiguous()}")
    print(f"  Data ptr: {t.data_ptr()}")
    print(f"  Storage bytes: {t.untyped_storage().nbytes()}")

t = torch.arange(12).reshape(3, 4)
print_tensor_info(t, "Original")
print_tensor_info(t.t(), "Transposed")
print_tensor_info(t[:, ::2], "Sliced")
```

## 10. Summary

### Key Concepts

1. **Storage is 1D**: Always a flat array of bytes, regardless of tensor shape
2. **Tensors are views**: Multiple tensors can share the same storage
3. **Strides define indexing**: How to convert multi-dimensional indices to storage indices
4. **Offset enables slicing**: Tensors can start at any position in storage
5. **Contiguity affects performance**: Contiguous tensors enable optimizations
6. **Device matters**: CPU and CUDA use different storage backends

### Storage Formula

To access element at index `[i₀, i₁, ..., iₙ]`:

```
storage_index = offset + Σ(iₖ * stride[k]) for k in 0..n
```

### Decision Tree

**Do I need to copy?**
- Changing dtype → Yes
- Changing device → Yes
- `clone()` → Yes
- `contiguous()` → Only if not contiguous
- Slicing → No (creates view)
- Transpose → No (creates view)
- `view()`/`reshape()` → Usually no (may require contiguous)

### Best Practices

1. **Minimize copies**: Use views when possible
2. **Check contiguity**: Call `contiguous()` only when needed
3. **Be aware of sharing**: `clone()` if you need independent storage
4. **Use appropriate dtype**: float32 for most ML, float16 for GPU training
5. **Understand strides**: Debug indexing issues by checking strides

The storage system enables PyTorch's efficiency - understanding it helps you write faster, more memory-efficient code.

## 11. C++ Implementation Details

PyTorch's Python tensors are thin wrappers around C++ objects. Understanding the C++ layer reveals how tensors actually work under the hood.

### 11.1 The C++ Architecture Stack

PyTorch's tensor implementation consists of several layers:

```
┌─────────────────────────────────┐
│   Python: torch.Tensor          │  ← Python wrapper
├─────────────────────────────────┤
│   C++: at::Tensor               │  ← ATen tensor (thin wrapper)
├─────────────────────────────────┤
│   C++: c10::TensorImpl          │  ← Core tensor implementation
├─────────────────────────────────┤
│   C++: c10::StorageImpl         │  ← Storage implementation
├─────────────────────────────────┤
│   C++: at::DataPtr              │  ← Smart pointer to raw data
├─────────────────────────────────┤
│   C++: at::Allocator            │  ← Memory allocation strategy
└─────────────────────────────────┘
```

### 11.2 StorageImpl: The C++ Storage Class

**Location**: `.venv/lib/python3.11/site-packages/torch/include/c10/core/StorageImpl.h`

The actual storage is implemented in C++ as `c10::StorageImpl`:

```cpp
struct StorageImpl : public c10::intrusive_ptr_target {
 private:
  at::DataPtr data_ptr_;          // Smart pointer to raw memory
  SymInt size_bytes_;             // Size in bytes
  bool resizable_;                // Can be resized?
  at::Allocator* allocator_;      // How to allocate/free memory

 public:
  // Get size in bytes
  size_t nbytes() const {
    return size_bytes_.as_int_unchecked();
  }

  // Get raw data pointer
  const void* data() const {
    return data_ptr_.get();
  }

  // Get mutable data pointer
  void* mutable_data() {
    return data_ptr_.mutable_get();
  }

  // Get the allocator
  at::Allocator* allocator() {
    return allocator_;
  }
};
```

**Key components**:
1. **`data_ptr_`**: Smart pointer managing the actual memory buffer
2. **`size_bytes_`**: Total size in bytes
3. **`allocator_`**: Responsible for malloc/free operations
4. **`resizable_`**: Whether storage can be resized

### 11.3 DataPtr: Smart Pointer with Device Info

**Location**: `.venv/lib/python3.11/site-packages/torch/include/c10/core/Allocator.h`

`DataPtr` is a unique pointer that knows its device:

```cpp
class DataPtr {
 private:
  c10::detail::UniqueVoidPtr ptr_;  // The actual pointer
  Device device_;                    // cpu, cuda, etc.

 public:
  void* get() const {
    return ptr_.get();
  }

  Device device() const {
    return device_;
  }

  // Move semantics (no copying!)
  DataPtr(DataPtr&&) = default;
  DataPtr(const DataPtr&) = delete;
};
```

**Why DataPtr?**
- Tracks which device owns the memory
- Ensures proper cleanup (calls device-specific free)
- Prevents copying (uses move semantics)

### 11.4 Allocators: CPU vs CUDA Memory

Different devices use different allocators:

#### CPU Allocator

```cpp
// Simplified CPU allocator
class CPUAllocator : public Allocator {
 public:
  DataPtr allocate(size_t nbytes) override {
    void* ptr = malloc(nbytes);  // Standard malloc
    return DataPtr(ptr, Device(DeviceType::CPU));
  }

  void deallocate(void* ptr) override {
    free(ptr);  // Standard free
  }
};
```

**CPU allocation**:
- Uses standard `malloc()` / `free()`
- Memory is in RAM (host memory)
- Accessible by CPU directly

#### CUDA Allocator

```cpp
// Simplified CUDA allocator
class CUDAAllocator : public Allocator {
 public:
  DataPtr allocate(size_t nbytes) override {
    void* ptr;
    cudaMalloc(&ptr, nbytes);  // CUDA malloc
    return DataPtr(ptr, Device(DeviceType::CUDA, 0));
  }

  void deallocate(void* ptr) override {
    cudaFree(ptr);  // CUDA free
  }
};
```

**CUDA allocation**:
- Uses `cudaMalloc()` / `cudaFree()`
- Memory is on GPU (device memory)
- CPU cannot directly access it
- Requires `cudaMemcpy()` to transfer data

### 11.5 Memory Allocation Flow

**Creating a CPU tensor**:

```python
t = torch.randn(10)
```

**What happens in C++**:

```cpp
// 1. Get CPU allocator
auto* allocator = GetAllocator(DeviceType::CPU);

// 2. Calculate bytes needed
size_t nbytes = 10 * sizeof(float);  // 40 bytes

// 3. Allocate memory
DataPtr data_ptr = allocator->allocate(nbytes);
// Internally: malloc(40)

// 4. Create StorageImpl
auto storage = make_intrusive<StorageImpl>(
    nbytes,
    std::move(data_ptr),
    allocator,
    /*resizable=*/true
);

// 5. Create TensorImpl
auto tensor_impl = make_intrusive<TensorImpl>(
    std::move(storage),
    /*key_set=*/...,
    /*dtype=*/torch::float32
);

// 6. Wrap in at::Tensor and return to Python
```

**Creating a CUDA tensor**:

```python
t = torch.randn(10, device='cuda')
```

**What happens in C++**:

```cpp
// 1. Get CUDA allocator
auto* allocator = GetAllocator(DeviceType::CUDA);

// 2. Allocate GPU memory
DataPtr data_ptr = allocator->allocate(40);
// Internally: cudaMalloc(&ptr, 40)

// 3. Rest is same as CPU, but DataPtr knows it's CUDA memory
```

### 11.6 CPU vs CUDA: Key Differences

| Aspect | CPU Storage | CUDA Storage |
|--------|-------------|--------------|
| **Allocator** | CPUAllocator | CUDAAllocator |
| **Allocation** | `malloc()` | `cudaMalloc()` |
| **Deallocation** | `free()` | `cudaFree()` |
| **Memory location** | RAM (host) | GPU VRAM (device) |
| **CPU access** | Direct | No (requires copy) |
| **GPU access** | No (requires copy) | Direct |
| **Data pointer** | Host address | Device address |

**Moving between devices**:

```python
cpu_t = torch.randn(10)          # malloc() in RAM
cuda_t = cpu_t.cuda()            # cudaMalloc() + cudaMemcpy()
cpu_again = cuda_t.cpu()         # malloc() + cudaMemcpy()
```

**What happens**:
```cpp
// cpu_t.cuda()
void* cuda_ptr;
cudaMalloc(&cuda_ptr, 40);                    // Allocate on GPU
cudaMemcpy(cuda_ptr, cpu_ptr, 40,            // Copy CPU → GPU
           cudaMemcpyHostToDevice);

// cuda_t.cpu()
void* cpu_ptr = malloc(40);                   // Allocate on CPU
cudaMemcpy(cpu_ptr, cuda_ptr, 40,            // Copy GPU → CPU
           cudaMemcpyDeviceToHost);
```

### 11.7 Memory Caching and Pooling

PyTorch doesn't always call malloc/cudaMalloc directly - it uses **memory pools**:

**CPU Caching Allocator**:
```cpp
class CPUCachingAllocator {
  std::map<size_t, std::vector<void*>> free_blocks;

  void* allocate(size_t size) {
    // Check if we have a cached block
    if (free_blocks[size].size() > 0) {
      void* ptr = free_blocks[size].back();
      free_blocks[size].pop_back();
      return ptr;  // Reuse!
    }
    return malloc(size);  // Allocate new
  }

  void free(void* ptr, size_t size) {
    free_blocks[size].push_back(ptr);  // Cache for reuse
  }
};
```

**Benefits**:
- Avoid expensive malloc/cudaMalloc calls
- Reduce fragmentation
- 10-100x faster for repeated allocations

**CUDA Caching**:
```python
import torch

# First allocation: calls cudaMalloc
t1 = torch.randn(1000, device='cuda')
del t1  # Doesn't call cudaFree! Memory cached

# Second allocation: reuses cached memory!
t2 = torch.randn(1000, device='cuda')  # Fast!
```

**Check CUDA memory**:
```python
print(torch.cuda.memory_allocated())     # Actually allocated
print(torch.cuda.memory_reserved())      # Cached (reserved)
torch.cuda.empty_cache()                 # Free cached memory
```

### 11.8 TensorImpl: The Core Tensor Class

**Location**: `.venv/lib/python3.11/site-packages/torch/include/c10/core/TensorImpl.h`

`TensorImpl` holds all tensor metadata:

```cpp
struct TensorImpl {
  // Storage
  c10::intrusive_ptr<StorageImpl> storage_;

  // Shape and strides
  SmallVector<int64_t, 5> sizes_;
  SmallVector<int64_t, 5> strides_;

  // Offset into storage
  int64_t storage_offset_;

  // Type information
  ScalarType dtype_;        // float32, int64, etc.
  DispatchKeySet key_set_;  // CPU, CUDA, autograd, etc.

  // Get element at multi-dimensional index
  void* data_at(ArrayRef<int64_t> indices) {
    int64_t offset = storage_offset_;
    for (size_t i = 0; i < indices.size(); i++) {
      offset += indices[i] * strides_[i];
    }
    return static_cast<char*>(storage_->data()) +
           offset * dtype_size();
  }
};
```

**Accessing data**:
```cpp
// Tensor: t[i, j, k]
// Storage index = offset + i*stride[0] + j*stride[1] + k*stride[2]
void* ptr = storage_->data();
size_t byte_offset = (offset + i*s0 + j*s1 + k*s2) * element_size;
float* element = static_cast<float*>(ptr) + byte_offset;
```

### 11.9 Python-C++ Bridge

**How Python tensors connect to C++**:

```python
import torch

t = torch.randn(2, 3)
print(type(t))           # <class 'torch.Tensor'>
print(t._cdata)          # C++ TensorImpl pointer
```

**The connection**:
```cpp
// Python torch.Tensor wraps C++ at::Tensor
class Tensor {  // Python wrapper
  at::Tensor tensor_;  // C++ tensor

  // Python calls forward to C++
  Tensor t() {
    return Tensor(tensor_.t());  // C++ transpose
  }
};

// at::Tensor wraps TensorImpl
class at::Tensor {
  c10::intrusive_ptr<TensorImpl> impl_;

  Tensor t() {
    // Create new TensorImpl with swapped strides
    auto new_impl = impl_->transpose(0, 1);
    return Tensor(std::move(new_impl));
  }
};
```

### 11.10 Memory Layout in C++

**Contiguous tensor in memory**:

```python
t = torch.arange(6).reshape(2, 3)
# [[0, 1, 2],
#  [3, 4, 5]]
```

**C++ memory structure**:
```
StorageImpl {
  data_ptr_ → [0, 1, 2, 3, 4, 5]  (24 bytes)
  size_bytes_ = 24
  allocator_ = CPUAllocator
}

TensorImpl {
  storage_ → StorageImpl (above)
  sizes_ = [2, 3]
  strides_ = [3, 1]
  storage_offset_ = 0
  dtype_ = kLong (int64)
}
```

**Non-contiguous tensor (transpose)**:

```python
t_T = t.t()  # [[0, 3],
             #  [1, 4],
             #  [2, 5]]
```

**C++ memory structure**:
```
StorageImpl {
  data_ptr_ → [0, 1, 2, 3, 4, 5]  (SAME storage!)
  size_bytes_ = 24
  allocator_ = CPUAllocator
}

TensorImpl {
  storage_ → StorageImpl (same pointer!)
  sizes_ = [3, 2]         ← Changed
  strides_ = [1, 3]       ← Changed (swapped)
  storage_offset_ = 0
  dtype_ = kLong
}
```

### 11.11 Reference Counting

C++ uses reference counting for memory management:

```cpp
// intrusive_ptr provides reference counting
c10::intrusive_ptr<StorageImpl> storage;

// Multiple tensors can share storage
auto t1_impl = make_tensor(storage);  // refcount = 2
auto t2_impl = make_tensor(storage);  // refcount = 3

// When last reference dies, storage is freed
t1_impl.reset();  // refcount = 2
t2_impl.reset();  // refcount = 1
storage.reset();  // refcount = 0 → free() called
```

**In Python**:
```python
t1 = torch.randn(10)
t2 = t1  # Share storage, refcount++

del t1   # refcount--
del t2   # refcount-- → 0 → C++ free() called
```

## 12. Summary: Python to C++ Mapping

| Python | C++ |
|--------|-----|
| `torch.Tensor` | `at::Tensor` wrapping `c10::TensorImpl*` |
| `tensor.untyped_storage()` | `c10::StorageImpl*` |
| `tensor.data_ptr()` | `StorageImpl::data_ptr_.get()` |
| `tensor.shape` | `TensorImpl::sizes_` |
| `tensor.stride()` | `TensorImpl::strides_` |
| `tensor.storage_offset()` | `TensorImpl::storage_offset_` |
| `tensor.device` | `DataPtr::device_` |
| `torch.randn(10)` | `CPUAllocator::allocate()` → `malloc()` |
| `tensor.cuda()` | `CUDAAllocator::allocate()` → `cudaMalloc()` + `cudaMemcpy()` |

The C++ implementation provides the performance-critical foundation while Python provides the user-friendly interface.
