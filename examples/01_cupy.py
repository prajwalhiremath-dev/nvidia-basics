"""
01_cupy.py — CuPy: GPU arrays + CUDA math libraries

CuPy is the entry point to almost every low-level NVIDIA math library.
When you call a CuPy function, it dispatches to the matching CUDA library:

  cp.matmul          →  cuBLAS  (sgemm / dgemm)
  cp.fft.*           →  cuFFT
  cp.random.*        →  cuRAND
  cupyx.scipy.sparse →  cuSPARSE
  cp.linalg.*        →  cuSolver
  cp.RawKernel       →  raw CUDA C kernel you write yourself

Run this file on any machine with a GPU and CuPy installed.
"""

import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as spla

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic GPU arrays
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("1. BASIC GPU ARRAYS")
print("=" * 60)

# Exactly like NumPy — all data lives on GPU memory
a = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
b = cp.zeros((3, 4), dtype=cp.float32)
c = cp.ones((3, 4), dtype=cp.float32)
r = cp.arange(12).reshape(3, 4).astype(cp.float32)

print(f"a        : {a}")
print(f"b shape  : {b.shape}")
print(f"r        :\n{r}")

# Move to / from CPU
cpu_array = cp.asnumpy(a)           # GPU → CPU
gpu_array = cp.asarray(cpu_array)   # CPU → GPU
print(f"Round-trip OK: {np.allclose(cpu_array, cp.asnumpy(gpu_array))}")

# GPU memory info
free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
print(f"\nGPU memory  — total: {total_bytes/1e9:.1f} GB  free: {free_bytes/1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# 2. cuBLAS — matrix multiply  (cp.matmul calls cublasSgemm internally)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. cuBLAS — MATRIX MULTIPLY")
print("=" * 60)

N = 4096
A_gpu = cp.random.randn(N, N, dtype=cp.float32)
B_gpu = cp.random.randn(N, N, dtype=cp.float32)

# Warm-up
_ = cp.matmul(A_gpu, B_gpu)
cp.cuda.stream.get_current_stream().synchronize()

t0 = time.perf_counter()
C_gpu = cp.matmul(A_gpu, B_gpu)          # cuBLAS SGEMM
cp.cuda.stream.get_current_stream().synchronize()
gpu_ms = (time.perf_counter() - t0) * 1000

A_cpu = cp.asnumpy(A_gpu)
B_cpu = cp.asnumpy(B_gpu)
t0 = time.perf_counter()
C_cpu = np.matmul(A_cpu, B_cpu)
cpu_ms = (time.perf_counter() - t0) * 1000

print(f"Matrix size : {N}x{N}  (float32)")
print(f"cuBLAS GPU  : {gpu_ms:.1f} ms")
print(f"NumPy  CPU  : {cpu_ms:.1f} ms")
print(f"Speedup     : {cpu_ms/gpu_ms:.1f}x")

# cuBLAS also powers: dot, inner, outer, tensordot, vdot
dot_result = cp.dot(A_gpu[0], B_gpu[:, 0])
print(f"cp.dot (row·col) result shape: {dot_result.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. cuFFT — Fast Fourier Transform
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. cuFFT — FAST FOURIER TRANSFORM")
print("=" * 60)

# 1-D FFT
N = 1 << 20   # 1 million points
t = cp.linspace(0, 1, N, dtype=cp.float32)
# Signal = 50 Hz + 120 Hz sine waves
signal = (cp.sin(2 * cp.pi * 50 * t) + 0.5 * cp.sin(2 * cp.pi * 120 * t))

t0 = time.perf_counter()
spectrum = cp.fft.rfft(signal)       # real-input FFT → calls cuFFT
cp.cuda.stream.get_current_stream().synchronize()
fft_ms = (time.perf_counter() - t0) * 1000

freqs = cp.fft.rfftfreq(N, d=1.0/N)
magnitudes = cp.abs(spectrum)
peak_freq = float(freqs[int(cp.argmax(magnitudes))])

print(f"Signal length : {N:,} samples")
print(f"cuFFT time    : {fft_ms:.2f} ms")
print(f"Peak freq     : {peak_freq:.0f} Hz  (expected 50)")

# 2-D FFT (image processing / convolution)
img = cp.random.randn(1024, 1024, dtype=cp.float32)
img_freq = cp.fft.fft2(img)         # 2D cuFFT
img_back = cp.fft.ifft2(img_freq).real
print(f"2D FFT roundtrip error: {float(cp.max(cp.abs(img_back - img))):.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. cuRAND — Random Number Generation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. cuRAND — RANDOM NUMBER GENERATION")
print("=" * 60)

N = 10_000_000

# All cp.random calls use cuRAND under the hood
uniform   = cp.random.rand(N, dtype=cp.float32)          # Uniform [0, 1)
normal    = cp.random.randn(N).astype(cp.float32)        # Standard normal
integers  = cp.random.randint(0, 100, size=N)            # Discrete uniform
exponential = cp.random.exponential(scale=2.0, size=N).astype(cp.float32)
poisson   = cp.random.poisson(lam=5.0, size=N)          # Poisson

print(f"Uniform   — mean: {float(uniform.mean()):.4f}  (want 0.5)")
print(f"Normal    — mean: {float(normal.mean()):.4f}  std: {float(normal.std()):.4f}")
print(f"Poisson   — mean: {float(poisson.mean()):.4f}  (want 5.0)")

# Seeding for reproducibility
cp.random.seed(42)
x1 = cp.random.randn(5)
cp.random.seed(42)
x2 = cp.random.randn(5)
print(f"Seeded reproducible: {cp.allclose(x1, x2)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. cuSPARSE — Sparse Matrix Operations
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. cuSPARSE — SPARSE MATRIX OPERATIONS")
print("=" * 60)

# Build a sparse matrix (CSR format) on GPU
#   cuSPARSE accelerates: SpMV, SpMM, sparse-dense products
n = 5000
density = 0.01
rng = np.random.default_rng(0)
row  = rng.integers(0, n, size=int(n * n * density))
col  = rng.integers(0, n, size=int(n * n * density))
data = rng.standard_normal(len(row)).astype(np.float32)

A_sparse = sp.csr_matrix(
    (cp.asarray(data), (cp.asarray(row), cp.asarray(col))),
    shape=(n, n)
)
x_vec = cp.random.randn(n, dtype=cp.float32)

t0 = time.perf_counter()
y = A_sparse @ x_vec                 # Sparse-matrix × dense-vector (cuSPARSE SpMV)
cp.cuda.stream.get_current_stream().synchronize()
sparse_ms = (time.perf_counter() - t0) * 1000

print(f"Sparse matrix: {n}x{n}  nnz={A_sparse.nnz:,}  density={density:.1%}")
print(f"SpMV time (cuSPARSE): {sparse_ms:.3f} ms")
print(f"Result vector norm  : {float(cp.linalg.norm(y)):.4f}")

# Sparse linear solve  (cuSPARSE + cuSolver)
b_vec = cp.random.randn(n, dtype=cp.float64).astype(cp.float32)
# Make diagonally dominant so the system is solvable
A_diag = A_sparse + sp.eye(n, format='csr') * 10.0
solution = spla.spsolve(A_diag.astype(cp.float64), b_vec.astype(cp.float64))
residual = float(cp.linalg.norm(A_diag.astype(cp.float64) @ solution - b_vec.astype(cp.float64)))
print(f"Sparse solve residual: {residual:.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. cuSolver — Dense Linear Algebra
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. cuSolver — DENSE LINEAR ALGEBRA")
print("=" * 60)

# cp.linalg calls cuSolver for decompositions and solvers
M = 2000
A_dense = cp.random.randn(M, M, dtype=cp.float32)
A_sym   = A_dense @ A_dense.T + cp.eye(M) * M  # Make positive-definite

# SVD  (cuSolver gesvd)
t0 = time.perf_counter()
U, S, Vt = cp.linalg.svd(A_dense[:200, :200], full_matrices=False)
cp.cuda.stream.get_current_stream().synchronize()
print(f"SVD (200x200)  time: {(time.perf_counter()-t0)*1000:.1f} ms  "
      f"top singular value: {float(S[0]):.2f}")

# QR decomposition  (cuSolver geqrf)
t0 = time.perf_counter()
Q, R = cp.linalg.qr(A_dense[:500, :500])
cp.cuda.stream.get_current_stream().synchronize()
print(f"QR  (500x500)  time: {(time.perf_counter()-t0)*1000:.1f} ms")

# Linear solve Ax=b  (cuSolver gesv)
b_rhs = cp.random.randn(M, dtype=cp.float32)
t0 = time.perf_counter()
x_sol = cp.linalg.solve(A_sym, b_rhs)
cp.cuda.stream.get_current_stream().synchronize()
residual = float(cp.linalg.norm(A_sym @ x_sol - b_rhs))
print(f"Solve ({M}x{M})  time: {(time.perf_counter()-t0)*1000:.1f} ms  "
      f"residual: {residual:.2e}")

# Eigenvalue decomposition  (cuSolver dsyevd for symmetric)
t0 = time.perf_counter()
eigenvalues, eigenvectors = cp.linalg.eigh(A_sym[:200, :200])
cp.cuda.stream.get_current_stream().synchronize()
print(f"Eigh (200x200) time: {(time.perf_counter()-t0)*1000:.1f} ms  "
      f"min eigenvalue: {float(eigenvalues[0]):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Custom CUDA Kernel with cp.RawKernel
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. CUSTOM CUDA KERNEL — cp.RawKernel")
print("=" * 60)

# Write raw CUDA C code as a Python string.
# This is compiled at runtime via NVRTC (NVIDIA Runtime Compilation).
cuda_code = r"""
extern "C" __global__
void vector_add_scaled(
    const float* a,
    const float* b,
    float*       c,
    float        alpha,
    int          n
) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = alpha * a[idx] + b[idx];
    }
}
"""

kernel = cp.RawKernel(cuda_code, "vector_add_scaled")

N = 1_000_000
alpha = cp.float32(2.5)
a_gpu = cp.random.randn(N, dtype=cp.float32)
b_gpu = cp.random.randn(N, dtype=cp.float32)
c_gpu = cp.empty(N, dtype=cp.float32)

threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block

kernel((blocks,), (threads_per_block,), (a_gpu, b_gpu, c_gpu, alpha, N))
cp.cuda.stream.get_current_stream().synchronize()

expected = float(alpha) * a_gpu + b_gpu
print(f"Custom kernel result matches CuPy: {cp.allclose(c_gpu, expected)}")
print(f"Max error: {float(cp.max(cp.abs(c_gpu - expected))):.2e}")

# More advanced kernel — parallel reduction (sum)
reduce_code = r"""
extern "C" __global__
void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (idx < n)       val += input[idx];
    if (idx + blockDim.x < n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
"""

reduce_kernel = cp.RawKernel(reduce_code, "reduce_sum")

data = cp.ones(1024, dtype=cp.float32)
block_sz = 256
n_blocks  = (len(data) + block_sz * 2 - 1) // (block_sz * 2)
partial   = cp.zeros(n_blocks, dtype=cp.float32)

reduce_kernel(
    (n_blocks,), (block_sz,),
    (data, partial, len(data)),
    shared_mem=block_sz * 4  # float32 = 4 bytes
)
total = float(partial.sum())
print(f"Parallel reduce sum: {total:.0f}  (expected {len(data)})")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CUDA Streams — async / overlapped execution
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. CUDA STREAMS — ASYNC / OVERLAPPED EXECUTION")
print("=" * 60)

# Streams let you run operations in parallel on the GPU.
# Without streams everything runs on the default stream (serialised).
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

X1 = cp.random.randn(2000, 2000, dtype=cp.float32)
X2 = cp.random.randn(2000, 2000, dtype=cp.float32)

with stream1:
    R1 = cp.matmul(X1, X1.T)      # runs on stream1

with stream2:
    R2 = cp.fft.fft2(X2)          # runs on stream2 concurrently

stream1.synchronize()
stream2.synchronize()

print(f"Stream1 result shape : {R1.shape}")
print(f"Stream2 result shape : {R2.shape}")
print("Both streams completed concurrently.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Unified Memory (managed memory accessible from CPU and GPU)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("9. UNIFIED / MANAGED MEMORY")
print("=" * 60)

# Managed memory is automatically migrated between CPU and GPU by the driver
mem = cp.cuda.alloc_managed(4 * 1024)   # 4 bytes × 1024 elements
arr = cp.ndarray(1024, dtype=cp.float32, memptr=mem)
arr[:] = cp.arange(1024, dtype=cp.float32)
cp.cuda.stream.get_current_stream().synchronize()

cpu_view = cp.asnumpy(arr)
print(f"First element via CPU view: {cpu_view[0]}")
print(f"Last  element via CPU view: {cpu_view[-1]}")

print("\n✓  01_cupy.py complete")
