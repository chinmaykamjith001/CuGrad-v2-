# CuGrad Benchmarks: CPU vs. GPU Training Performance

This document analyzes the performance of CuGrad's custom CUDA backend against a
single-threaded CPU baseline of the same autograd engine. Both implementations share
identical model code, weight initialization (seeded), data, and hyperparameters —
the CPU version (`engine.cpp`) simply replaces `cudaMallocManaged` tensors and the 14
CUDA kernels with `std::vector`-backed tensors and plain nested-loop math, so the
comparison isolates the effect of the GPU backend itself.

## 1. Setup

| | |
|---|---|
| Hardware | NVIDIA T4 GPU (Google Colab) vs. single CPU thread |
| Model | MLP, 512 → 256 → 64 → 1, ReLU hidden layers |
| Task | Synthetic binary classification (only 1 of 512 features is predictive) |
| Dataset size | 50,000 samples |
| Epochs | 10 (fixed across all runs) |
| Optimizer | Plain SGD, lr = 0.03 |
| Loss | MSE |
| Variable | Batch size, swept from 64 → 50,000 |

Both engines build the same computational graph and run the same 14-operation set
(add, sub, mul, div, matmul, relu, sgd update); the GPU version launches each as a
CUDA kernel with a grid-stride loop, the CPU version runs it as a straight loop.

## 2. Headline result

Across the batch-size sweep, the GPU backend was **13×–197× faster** than the CPU
baseline, with the exact multiplier depending heavily on batch size (see §4). At
batch size 4096 — the default used in the standalone CPU benchmark script — the
speedup was **~144×**.

<img width="1275" height="825" alt="3_cpu_vs_gpu" src="https://github.com/user-attachments/assets/a4c18e0b-f22a-47e7-a097-6bd9588ff71f" />


## 3. Raw training time vs. batch size

<img width="1200" height="825" alt="1_cpu_times" src="https://github.com/user-attachments/assets/7b376b3e-b50d-4097-9407-b4af8d21e992" />


CPU time trends *upward* with batch size even though total FLOPs for a full epoch
are fixed (50,000 samples get processed once per epoch regardless of how they're
batched). This is a cache-locality effect, not a FLOP-count effect: the naive
triple-loop matmul (`engine.cpp`) has a working set that grows with batch size (M),
so larger batches push more of the matmul out of L1/L2 cache and increase memory
stalls per FLOP.

<img width="1200" height="825" alt="2_gpu_times" src="https://github.com/user-attachments/assets/fcd834fc-d1b9-4432-aa2d-df2a5567d8a0" />


GPU time drops sharply as batch size increases, then plateaus (and slightly dips)
in the 2,048–32,768 range. This is the classic **kernel-launch-overhead** story: at
batch size 64, an epoch launches ~780 tiny kernels (50000/64 ≈ 781 batches × ~14
kernels), each dominated by launch latency rather than compute. As batch size grows,
fewer, larger kernels amortize that fixed launch cost, until the GPU is
compute-bound and the curve flattens.

## 4. Speedup curve

<img width="1200" height="825" alt="4_speedup" src="https://github.com/user-attachments/assets/60b8d15c-310d-431c-a764-bccc7d6454d4" />


Speedup rises steeply through the small-batch regime (13× → 88× between batch 64
and 1024) as launch overhead stops dominating the GPU side, then continues rising
more slowly toward ~190–197× as batch size approaches the full dataset size. This
is the product of two independent trends compounding: CPU time going up, GPU time
going down.

## 5. Comparison Against PyTorch

While the previous sections evaluate CuGrad relative to its own CPU implementation, it is also useful to compare against an established deep learning framework.

To provide this context, the same network architecture, dataset, optimizer, epoch count, and batch-size sweep were implemented in PyTorch and executed on the same NVIDIA T4 GPU. Although the underlying CUDA kernels and implementation details differ substantially, this comparison gives a practical reference point for where CuGrad currently stands relative to a production-grade framework.

<img width="1600" height="1000" alt="Code_Generated_Image(2)" src="https://github.com/user-attachments/assets/6b0272aa-ee5f-4418-9e52-eb6da01571cd" />

PyTorch exhibits the expected behavior of a mature CUDA backend. Execution time decreases rapidly as batch size increases, before flattening once kernel launch overhead becomes negligible and computation becomes throughput-limited. The curve closely follows the qualitative behavior observed for CuGrad, suggesting that both implementations are governed by the same underlying GPU execution characteristics.

<img width="1600" height="1000" alt="Code_Generated_Image(1)" src="https://github.com/user-attachments/assets/494d4725-eae7-4a23-8570-40f22aee565c" />

The direct comparison shows that PyTorch consistently outperforms CuGrad across the entire batch-size range. This is expected, as PyTorch benefits from years of engineering effort, highly optimized CUDA kernels, kernel fusion, memory pooling, asynchronous execution, and vendor-tuned libraries such as cuBLAS and cuDNN.

By contrast, CuGrad intentionally uses straightforward CUDA implementations of tensor operations to emphasize correctness, transparency, and educational value rather than peak hardware utilization.

<img width="1600" height="1000" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/801b44b5-8d4f-42ca-b2c0-4647922e71ce" />

Viewing both implementations on logarithmic axes highlights that they exhibit nearly identical scaling behavior despite differing absolute performance. Both curves decrease rapidly at small batch sizes before approaching a plateau at larger batches, indicating that CuGrad captures the same fundamental GPU scaling trend as PyTorch even though its constant-factor overhead remains larger.

Overall, these results demonstrate that CuGrad behaves similarly to modern GPU frameworks while leaving substantial room for future optimization. Potential improvements include tiled matrix multiplication, kernel fusion, CUDA streams, optimized memory allocation, shared-memory caching, and integration with vendor-optimized libraries.

## 6. Loss vs. batch size — and why it's *not* a hardware effect

<img width="1200" height="825" alt="5_loss" src="https://github.com/user-attachments/assets/bbe06f58-c24f-47cb-9f59-272a4bd4e660" />


Loss increases monotonically with batch size. It's tempting to read this as "GPU
training converges worse," but that's not what's happening — **epochs are fixed at
10, so a larger batch size directly means fewer gradient updates per epoch**
(`batches = total_samples / batch_size`). At batch 64 the model takes ~781 SGD
steps per epoch; at batch 50,000 it takes exactly 1. This is a step-budget artifact
of the experiment design, not a property of the GPU backend — the same curve would
appear if the CPU engine were run alone across these batch sizes. Worth stating
explicitly in the write-up so it doesn't read as a hardware tradeoff.

Additionally, since lower batch sizes reduce a number of elements in a batch, there is less variance and therefore lower loss value. Does not indicate any sort of improvement in and of itself; simply a side-effect of lower batch sizes.

## 7. Practical batch-size tradeoff

<img width="1275" height="900" alt="7_pareto_speedup_loss" src="https://github.com/user-attachments/assets/9f779587-b4db-4962-a339-36c36f07f330" />


Plotting speedup directly against loss shows the frontier bending hard after
batch ≈1024–2048: past that point you're trading large amounts of convergence
quality for comparatively small additional speed.

<img width="1275" height="825" alt="8_efficiency_score" src="https://github.com/user-attachments/assets/dae4e401-841a-4059-b290-f88c2da7183a" />


A simple speedup/loss ratio flags small batches (64) as most "efficient" by this
metric — but that's mostly because loss is near zero there, not because it's
actually the best operating point. Given §5, the right fix if you want a genuinely
fair batch-size comparison is to **scale learning rate and/or number of epochs with
batch size** (e.g., linear LR scaling, or fix total gradient steps instead of total
epochs) so accuracy isn't confounded with step count. Flagging this as a limitation
is more credible than presenting the ratio as a real answer.

<img width="1500" height="825" alt="9_marginal_speedup_gain" src="https://github.com/user-attachments/assets/e3258e86-e32a-4d8c-ae48-c1423e23868a" />


Diminishing returns per doubling: big jumps early, near-zero or slightly negative
gains at the top end.

## 8. Throughput

<img width="1275" height="825" alt="6b_throughput" src="https://github.com/user-attachments/assets/b987f134-45cb-4362-a4d1-6e46e8fec356" />


Reframes the same data as samples/sec, which is arguably the more standard metric
for reporting training capacity.

## 9. Limitations

- **Single trial per batch size** — no error bars; the CPU dip at batch 8192 (746s,
  lower than neighbors) and the GPU dip at batch 32768 (4.08s, lower than
  neighbors) both look like measurement noise and should be rerun before being
  presented as trend.
- **CPU baseline is single-threaded and unoptimized** (naive triple-loop matmul,
  no SIMD/BLAS). This makes for a clean apples-to-apples comparison of "custom
  kernels vs. no kernels," but it's not a fair fight against a properly optimized
  CPU baseline (e.g., OpenMP + a BLAS library). Worth stating this explicitly so
  the 144–197× number isn't read as "GPU beats best-effort CPU" — it's "custom
  CUDA beats naive single-thread CPU."
- **Loss confounded with step count** across the batch sweep (§5) — not yet
  controlled for.
- **Unified memory (`cudaMallocManaged`)** is used for simplicity/correctness
  rather than performance; explicit host/device memory management with pinned
  memory would likely shift the small-batch numbers further in the GPU's favor by
  reducing page-fault overhead.

## 10. Suggested next steps

- Rerun batch 8192 and 32768 to confirm or replace the outlier points.
- Add an OpenMP-parallelized CPU baseline as a second comparison point.
- Control for gradient steps (not just epochs) across the batch sweep to get a
  clean convergence comparison.
- Profile with `nsys`/`nvprof` to confirm the kernel-launch-overhead explanation
  in §3 rather than inferring it from timing shape alone.
