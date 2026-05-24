"""
03_cuml.py — cuML: GPU Machine Learning

cuML is GPU-accelerated scikit-learn.
Every algorithm runs entirely on the GPU — the API is intentionally identical
to sklearn so you can swap `sklearn` → `cuml` with almost no code changes.

Works directly with cuDF DataFrames and CuPy arrays: no conversion to NumPy.

Algorithms covered here:
  cuml.linear_model   — LinearRegression, LogisticRegression, Ridge, Lasso
  cuml.ensemble       — RandomForestClassifier, RandomForestRegressor
  cuml.cluster        — KMeans, DBSCAN
  cuml.decomposition  — PCA, TruncatedSVD
  cuml.manifold       — UMAP (t-SNE equivalent, much faster)
  cuml.svm            — SVC, SVR
  cuml.preprocessing  — StandardScaler, MinMaxScaler
  cuml.model_selection— train_test_split
  cuml.metrics        — accuracy_score, r2_score
"""

import time
import numpy as np
import cupy as cp
import cudf

# cuML sub-modules
from cuml.linear_model  import LinearRegression, LogisticRegression, Ridge, Lasso
from cuml.ensemble      import RandomForestClassifier, RandomForestRegressor
from cuml.cluster       import KMeans, DBSCAN
from cuml.decomposition import PCA, TruncatedSVD
from cuml.manifold      import UMAP
from cuml.svm           import SVC
from cuml.preprocessing import StandardScaler, MinMaxScaler
from cuml.model_selection import train_test_split
from cuml.metrics       import accuracy_score, r2_score

import cuml
print(f"cuML version: {cuml.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper — print timing comparison
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(label, fn):
    t0 = time.perf_counter()
    result = fn()
    cp.cuda.stream.get_current_stream().synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label}: {ms:.1f} ms")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(0)
N, D = 100_000, 20     # 100 k samples, 20 features

# Regression dataset: y = X·w + noise
true_w = rng.standard_normal(D).astype(np.float32)
X_np   = rng.standard_normal((N, D)).astype(np.float32)
y_reg  = X_np @ true_w + rng.standard_normal(N).astype(np.float32) * 0.1

# Classification dataset (5 classes)
X_cls  = rng.standard_normal((N, D)).astype(np.float32)
y_cls  = (X_cls[:, 0] * 2 + X_cls[:, 1] > 0).astype(np.int32)   # binary

# Convert to cuDF for zero-copy ML pipelines
X_cudf  = cudf.DataFrame(X_np, columns=[f"f{i}" for i in range(D)])
yr_cudf = cudf.Series(y_reg, name="target")
yc_cudf = cudf.Series(y_cls, name="label")

# Train/test split on GPU
X_train, X_test, yr_train, yr_test = train_test_split(
    X_cudf, yr_cudf, test_size=0.2, random_state=42
)
_, __, yc_train, yc_test = train_test_split(
    X_cudf, yc_cudf, test_size=0.2, random_state=42
)

print(f"Dataset: {N:,} samples × {D} features")
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Linear Regression  (cuBLAS under the hood)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("1. LINEAR REGRESSION")
print("=" * 60)

lr = LinearRegression(algorithm="svd", fit_intercept=True)
benchmark("fit", lambda: lr.fit(X_train, yr_train))
preds = lr.predict(X_test)
r2 = r2_score(yr_test, preds)
print(f"  R² score : {float(r2):.4f}  (1.0 = perfect)")

# Weight recovery check
learned_w = cp.asnumpy(lr.coef_)
true_w_sorted = np.sort(np.abs(true_w))
learned_sorted = np.sort(np.abs(learned_w))
print(f"  Correlation of true vs learned weights: "
      f"{np.corrcoef(true_w, learned_w)[0,1]:.4f}")

# Ridge (L2 regularisation)
ridge = Ridge(alpha=1.0, fit_intercept=True)
benchmark("Ridge fit", lambda: ridge.fit(X_train, yr_train))
r2_ridge = r2_score(yr_test, ridge.predict(X_test))
print(f"  Ridge R²: {float(r2_ridge):.4f}")

# Lasso (L1 regularisation — produces sparse coefficients)
lasso = Lasso(alpha=0.01, fit_intercept=True)
benchmark("Lasso fit", lambda: lasso.fit(X_train, yr_train))
r2_lasso = r2_score(yr_test, lasso.predict(X_test))
print(f"  Lasso R²: {float(r2_lasso):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. LOGISTIC REGRESSION")
print("=" * 60)

log_reg = LogisticRegression(max_iter=1000, C=1.0, fit_intercept=True)
benchmark("fit", lambda: log_reg.fit(X_train, yc_train))
cls_preds = log_reg.predict(X_test)
acc = accuracy_score(yc_test, cls_preds)
print(f"  Accuracy: {float(acc):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. RANDOM FOREST")
print("=" * 60)

# Classifier
rf_cls = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    n_streams=4,         # cuML uses multiple GPU streams for tree building
    random_state=42,
)
benchmark("RF Classifier fit", lambda: rf_cls.fit(X_train, yc_train))
rf_preds = rf_cls.predict(X_test)
rf_acc = accuracy_score(yc_test, rf_preds)
print(f"  RF Accuracy: {float(rf_acc):.4f}")

# Regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42,
)
benchmark("RF Regressor fit", lambda: rf_reg.fit(X_train, yr_train))
rf_r2 = r2_score(yr_test, rf_reg.predict(X_test))
print(f"  RF R²: {float(rf_r2):.4f}")

# Feature importances (which features matter most)
importances = cp.asnumpy(rf_cls.feature_importances_)
top3 = np.argsort(importances)[::-1][:3]
print(f"  Top-3 important features: {[f'f{i}' for i in top3]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. KMeans Clustering
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. KMEANS CLUSTERING")
print("=" * 60)

# Build a dataset with 5 clear clusters
centers = rng.standard_normal((5, D)).astype(np.float32) * 5
cluster_data = np.vstack([
    rng.standard_normal((N // 5, D)).astype(np.float32) + centers[i]
    for i in range(5)
])
X_cluster = cudf.DataFrame(cluster_data)

kmeans = KMeans(
    n_clusters=5,
    max_iter=300,
    tol=1e-4,
    init="scalable-k-means++",
    random_state=42,
)
benchmark("KMeans fit", lambda: kmeans.fit(X_cluster))
cluster_labels = kmeans.labels_
inertia = float(kmeans.inertia_)
print(f"  Inertia (lower = tighter clusters): {inertia:.2f}")
print(f"  Unique labels found: {sorted(cluster_labels.unique().to_arrow().to_pylist())}")

# Predict new points
new_pts = cudf.DataFrame(rng.standard_normal((10, D)).astype(np.float32))
assigned = kmeans.predict(new_pts)
print(f"  New point assignments: {assigned.to_arrow().to_pylist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DBSCAN — Density-Based Clustering (finds irregular shapes)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. DBSCAN CLUSTERING")
print("=" * 60)

# DBSCAN doesn't need k upfront — it discovers clusters automatically
dbscan = DBSCAN(eps=1.2, min_samples=5, metric="euclidean")
small_cluster = cudf.DataFrame(cluster_data[:5000])  # subset for speed
benchmark("DBSCAN fit", lambda: dbscan.fit(small_cluster))

labels = dbscan.labels_
n_clusters  = int((labels >= 0).sum() > 0)
n_noise     = int((labels == -1).sum())
unique_lbls = len(labels.unique())
print(f"  Unique cluster labels: {unique_lbls}  (−1 = noise)")
print(f"  Noise points: {n_noise}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA — Principal Component Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. PCA — PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)

pca = PCA(n_components=10, svd_solver="full", random_state=42)
benchmark("PCA fit", lambda: pca.fit(X_cudf))
X_reduced = pca.transform(X_cudf)

explained = cp.asnumpy(pca.explained_variance_ratio_)
print(f"  Input  shape: {X_cudf.shape}")
print(f"  Output shape: {X_reduced.shape}")
print(f"  Variance explained by top-3 PCs: "
      f"{explained[:3].sum()*100:.1f}%")

# Inverse transform (reconstruct original)
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = float(
    cp.linalg.norm(
        cp.asarray(X_cudf.values) - cp.asarray(X_reconstructed.values)
    )
) / N
print(f"  Reconstruction error per sample: {reconstruction_error:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. UMAP — Dimensionality Reduction for Visualization
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. UMAP — NON-LINEAR DIMENSIONALITY REDUCTION")
print("=" * 60)

# UMAP reduces high-dimensional data to 2D/3D for visualization.
# cuML UMAP is 100-1000x faster than the CPU version.
umap = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)
X_small = cudf.DataFrame(cluster_data[:10_000])  # 10k points
benchmark("UMAP fit_transform", lambda: umap.fit_transform(X_small))
embedding = umap.transform(X_small)
print(f"  Input  shape: {X_small.shape}")
print(f"  Embedding shape: {embedding.shape}  (ready for scatter plot)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Support Vector Machine
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. SVC — SUPPORT VECTOR CLASSIFICATION")
print("=" * 60)

# Use a smaller subset — SVM is O(n²) memory
N_svm = 10_000
X_svm   = cudf.DataFrame(X_np[:N_svm])
y_svm   = cudf.Series(y_cls[:N_svm].astype(np.int32))
Xtr, Xte, ytr, yte = train_test_split(X_svm, y_svm, test_size=0.2, random_state=0)

svc = SVC(kernel="rbf", C=1.0, gamma="scale")
benchmark("SVC fit", lambda: svc.fit(Xtr, ytr))
svc_preds = svc.predict(Xte)
svc_acc = accuracy_score(yte, svc_preds)
print(f"  SVC Accuracy (RBF kernel): {float(svc_acc):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("9. PREPROCESSING — SCALERS")
print("=" * 60)

# StandardScaler — zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cudf)
col0_mean = float(X_scaled["f0"].mean())
col0_std  = float(X_scaled["f0"].std())
print(f"  After StandardScaler — mean: {col0_mean:.4f}  std: {col0_std:.4f}")

# MinMaxScaler — rescale to [0, 1]
mm_scaler = MinMaxScaler(feature_range=(0, 1))
X_mm = mm_scaler.fit_transform(X_cudf)
col0_min = float(X_mm["f0"].min())
col0_max = float(X_mm["f0"].max())
print(f"  After MinMaxScaler — min: {col0_min:.4f}  max: {col0_max:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Full sklearn-compatible Pipeline (cuML → matches sklearn pattern exactly)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("10. FULL GPU PIPELINE: SCALE → PCA → CLASSIFY")
print("=" * 60)

t0 = time.perf_counter()

# Step 1: Scale
sc = StandardScaler()
X_s = sc.fit_transform(X_train)

# Step 2: Reduce
p = PCA(n_components=10, random_state=0)
X_p = p.fit_transform(X_s)

# Step 3: Classify
clf = LogisticRegression(max_iter=500)
clf.fit(X_p, yc_train)

# Test
X_te_s = sc.transform(X_test)
X_te_p = p.transform(X_te_s)
pipe_acc = float(accuracy_score(yc_test, clf.predict(X_te_p)))

cp.cuda.stream.get_current_stream().synchronize()
pipe_ms = (time.perf_counter() - t0) * 1000

print(f"  Pipeline (Scale→PCA→LogReg) on {len(X_train):,} samples")
print(f"  Time   : {pipe_ms:.1f} ms")
print(f"  Test accuracy: {pipe_acc:.4f}")

print("\n✓  03_cuml.py complete")
