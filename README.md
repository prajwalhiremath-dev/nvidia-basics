# NVIDIA Libraries Tour

A hands-on learning repo covering the NVIDIA RAPIDS ecosystem and CUDA math libraries.
Every `.py` file mirrors a section in the master Colab notebook.

---

## Library Map

```
Your Code
    │
    ├── CuPy  ─────────────────────────────────────────────────
    │         GPU-accelerated NumPy.  Under the hood it calls:
    │           cuBLAS   (matrix multiply, triangular solve)
    │           cuFFT    (Fast Fourier Transform)
    │           cuRAND   (random number generation)
    │           cuSPARSE (sparse matrix ops)
    │           cuSolver (LU / QR / SVD decomposition)
    │         You can also write raw CUDA kernels via RawKernel.
    │
    ├── cuDF  ─────────────────────────────────────────────────
    │         GPU-accelerated pandas.  cuIO is built in:
    │           read_csv / read_parquet / read_json / read_orc
    │         Data never leaves the GPU during the pipeline.
    │
    ├── cuML  ─────────────────────────────────────────────────
    │         GPU-accelerated scikit-learn.
    │           Regression, Classification, Clustering,
    │           Dimensionality Reduction, Preprocessing.
    │         Works directly with cuDF DataFrames — no conversion.
    │
    ├── cuGraph ────────────────────────────────────────────────
    │         GPU-accelerated graph analytics.
    │           PageRank, BFS, SSSP, Louvain, Triangle Count,
    │           Connected Components, Jaccard Similarity.
    │         Reads cuDF edge-list DataFrames natively.
    │
    └── RMM  ───────────────────────────────────────────────────
              RAPIDS Memory Manager — the shared GPU memory pool
              used by all RAPIDS libraries.
```

---

## Files

| File | What you learn |
|------|---------------|
| `examples/01_cupy.py` | GPU arrays, cuBLAS/cuFFT/cuRAND/cuSPARSE via CuPy, custom CUDA kernels, streams |
| `examples/02_cudf.py` | GPU DataFrames, cuIO (CSV/Parquet/JSON), groupby, merge, string ops |
| `examples/03_cuml.py` | Linear/Logistic Regression, Random Forest, KMeans, DBSCAN, PCA, UMAP on GPU |
| `examples/04_cugraph.py` | PageRank, BFS, SSSP, Louvain community detection, triangle count |
| `nvidia_libraries_tour.ipynb` | Master Colab notebook — runs everything in one place |

---

## Quickstart in Colab

1. **Runtime → Change runtime type → GPU** (T4 or A100)
2. Upload this repo zip or clone it
3. Open `nvidia_libraries_tour.ipynb`
4. Run **Cell 1** — it installs all RAPIDS libraries automatically

---

## Install (standalone)

```bash
# CUDA 12.x  (what Colab uses)
pip install cupy-cuda12x \
    cudf-cu12==26.02.* \
    cuml-cu12==26.02.* \
    cugraph-cu12==26.02.* \
    --extra-index-url=https://pypi.nvidia.com
```

---

## NVIDIA Library Cheat-Sheet

| Library | Replaces | Key import |
|---------|----------|-----------|
| CuPy | NumPy | `import cupy as cp` |
| cuBLAS | BLAS (CPU) | Automatic via `cp.matmul` |
| cuFFT | SciPy FFT | Automatic via `cp.fft` |
| cuRAND | NumPy random | Automatic via `cp.random` |
| cuSPARSE | SciPy sparse | `import cupyx.scipy.sparse as sp` |
| cuSolver | SciPy linalg | Automatic via `cp.linalg` |
| cuDF | pandas | `import cudf` |
| cuIO | pandas IO | Built into cuDF (`cudf.read_csv` etc.) |
| cuML | scikit-learn | `import cuml` |
| cuGraph | NetworkX | `import cugraph` |
| RMM | — | `import rmm` |
