# CuGrad

A C++/CUDA deep learning engine built from scratch — automatic differentiation,
GPU tensor operations, and multilayer perceptrons, with no external ML
libraries. Every operator, from tensor broadcasting to SGD, is backed by a
hand-written CUDA kernel.

<img width="1275" height="825" alt="3_cpu_vs_gpu" src="https://github.com/user-attachments/assets/8a144e2d-4a76-4bbb-8dda-a1f4db948376" />


## Highlights

- Reverse-mode autodiff via a dynamically built computational graph (topological
  sort + `_backward` closures per tensor, in the style of micrograd)
- **14 custom CUDA kernels** covering elementwise ops (add/sub/mul/div, forward
  + backward), matrix multiplication (forward + 2 backward kernels), ReLU
  (forward + backward), and SGD parameter updates
- Tensor broadcasting for row-vector bias addition
- Grid-stride loop kernels + `atomicAdd`-based gradient accumulation for
  variable-size tensors
- Unified memory (`cudaMallocManaged`) tensors — no manual host/device copies
- Up to **197× faster** than a single-threaded CPU baseline of the same engine
  (NVIDIA T4, see [BENCHMARKS.md](./BENCHMARKS.md))

## Why

Most ML frameworks hide the backward pass and GPU dispatch behind layers of
abstraction. This project builds both from the ground up — starting from raw
`float*` buffers and hand-rolled kernels — to understand exactly what
autograd and GPU-accelerated training are doing underneath a framework like
PyTorch.

## Architecture

```
Tensor                 owning float* data/grad (managed memory), _prev graph
                        edges, _backward closure, shape/stride bookkeeping

Operators               operator+, operator-, operator*, operator/, matmul,
                        relu, sum, mean — each builds a graph node and
                        launches a forward kernel; backward() replays the
                        graph in reverse topological order, launching each
                        node's stored backward kernel

Layer / MLP             thin C++ wrappers composing Tensor ops into a
                        feedforward network (He-initialized weights, ReLU
                        hidden layers)

sgd_update               in-place parameter update kernel
```

All backward kernels launch on the default CUDA stream, so kernel ordering
and data dependencies are preserved without manual synchronization — the
engine only calls `cudaDeviceSynchronize()` where the **host** needs to read
a value (e.g., computing `mean()`/`sum()` for the loss, or after training
completes for timing).

## Build & run

Requires the CUDA toolkit (for the GPU version) and a CUDA-capable GPU.

```bash
# GPU version
nvcc -DMAIN_ACTIVE -O3 -arch=sm_75 engine.cu -o engine_gpu
./engine_gpu

# CPU baseline (for comparison — no CUDA required)
g++ -O3 engine.cpp -o engine_cpu
./engine_cpu
```

`-arch=sm_75` targets a T4 GPU; adjust for your hardware (e.g. `sm_86` for
Ampere, `sm_90` for Hopper).

## Benchmarks

Trained a 512→256→64→1 MLP on a synthetic binary classification task across
batch sizes from 64 to 50,000. GPU training was 13×–197× faster than the CPU
baseline depending on batch size (144× at batch size 4096).

<img width="1200" height="825" alt="4_speedup" src="https://github.com/user-attachments/assets/d8ae2bb4-9e11-456e-8bb9-dd9d78df3e3f" />

Additionally, in cretain situations, comes as close as nearly 40% of PyTorch (equivalent script implemented with PyTorch with identical configuration)

<img width="1600" height="1000" alt="Code_Generated_Image(1)" src="https://github.com/user-attachments/assets/365b86c1-c33a-462e-b823-09f2a27a34e4" />





Full methodology, all benchmark graphs, and discussion of limitations
(including a step-count confound in the loss-vs-batch-size results) are in
[BENCHMARKS.md](./BENCHMARKS.md).

## Status

Ongoing solo project, started September 2025. Currently supports dense
feedforward networks trained with plain SGD; no convolutional layers,
adaptive optimizers, or multi-GPU support yet.


