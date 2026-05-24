"""
02_cudf.py — cuDF + cuIO: GPU DataFrames

cuDF is GPU-accelerated pandas.  cuIO is the I/O layer built into cuDF.
Data is loaded straight onto GPU memory — no CPU detour.

NVIDIA libraries used here:
  cuDF   — DataFrame operations (filter, groupby, merge, sort, string ops)
  cuIO   — Reading/writing CSV, Parquet, JSON, ORC directly to GPU memory
  RMM    — Memory pool that cuDF uses internally
"""

import io
import time
import numpy as np
import cupy as cp
import cudf
import cudf.io

print(f"cuDF version: {cudf.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DataFrame Creation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("1. DATAFRAME CREATION")
print("=" * 60)

# From a Python dict — same API as pandas
df = cudf.DataFrame({
    "name"   : ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age"    : [25, 30, 35, 28, 22],
    "salary" : [70_000, 90_000, 85_000, 60_000, 55_000],
    "dept"   : ["Eng", "Eng", "HR", "Sales", "Eng"],
})
print(df)
print(f"\ndtype per column:\n{df.dtypes}")

# From NumPy arrays — data copied to GPU
N = 1_000_000
rng = np.random.default_rng(42)
big_df = cudf.DataFrame({
    "x"      : rng.standard_normal(N).astype(np.float32),
    "y"      : rng.standard_normal(N).astype(np.float32),
    "label"  : rng.integers(0, 5, N).astype(np.int32),
    "weight" : rng.uniform(0, 1, N).astype(np.float32),
})
print(f"\nLarge DataFrame: {len(big_df):,} rows, {len(big_df.columns)} columns")
print(big_df.head())

# From CuPy arrays — zero-copy, stays on GPU
gpu_arr = cp.arange(10, dtype=cp.float32)
series = cudf.Series(gpu_arr)
print(f"\nSeries from CuPy array: {series.values}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. cuIO — Read / Write (CSV, Parquet, JSON)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. cuIO — READ / WRITE")
print("=" * 60)

# --- CSV ---
csv_data = """id,product,price,quantity
1,Widget,9.99,100
2,Gadget,24.99,50
3,Doohickey,4.49,200
4,Thingamajig,14.99,75
5,Whatchamacallit,29.99,30
"""

# Read CSV directly to GPU (cuIO)
t0 = time.perf_counter()
csv_df = cudf.read_csv(io.StringIO(csv_data))
print(f"read_csv time : {(time.perf_counter()-t0)*1000:.2f} ms")
print(csv_df)

# Write CSV back
csv_df.to_csv("/tmp/output.csv", index=False)
print("Written to /tmp/output.csv")

# --- Parquet ---
# Parquet is columnar — much faster to read/write large data on GPU
big_df.to_parquet("/tmp/big_data.parquet", index=False)

t0 = time.perf_counter()
parquet_df = cudf.read_parquet("/tmp/big_data.parquet")
parquet_ms = (time.perf_counter() - t0) * 1000
print(f"\nParquet read  : {len(parquet_df):,} rows in {parquet_ms:.1f} ms")

# --- JSON ---
json_data = '[{"a":1,"b":2},{"a":3,"b":4},{"a":5,"b":6}]'
json_df = cudf.read_json(io.StringIO(json_data))
print(f"\nJSON DataFrame:\n{json_df}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Filtering and Slicing
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. FILTERING AND SLICING")
print("=" * 60)

# Boolean mask filter
high_earners = df[df["salary"] > 70_000]
print("High earners (salary > 70k):")
print(high_earners)

# Multi-condition
eng_seniors = big_df[(big_df["label"] == 0) & (big_df["x"] > 0)]
print(f"\nLabel=0 and x>0: {len(eng_seniors):,} rows")

# Column selection
subset = big_df[["x", "label"]]
print(f"Column subset shape: {subset.shape}")

# isin filter
dept_filter = df[df["dept"].isin(["Eng", "HR"])]
print(f"\nEng + HR dept rows: {len(dept_filter)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GroupBy Aggregations
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. GROUPBY AGGREGATIONS")
print("=" * 60)

# Single-column groupby
dept_stats = df.groupby("dept")["salary"].agg(["mean", "min", "max", "count"])
print("Salary stats by department:")
print(dept_stats)

# Multi-column aggregation on large data
t0 = time.perf_counter()
label_agg = big_df.groupby("label").agg({
    "x"      : ["mean", "std"],
    "y"      : ["mean", "std"],
    "weight" : "sum",
})
agg_ms = (time.perf_counter() - t0) * 1000
print(f"\nGroupBy on {len(big_df):,} rows took {agg_ms:.1f} ms")
print(label_agg)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Merge / Join
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. MERGE / JOIN")
print("=" * 60)

orders = cudf.DataFrame({
    "order_id"  : [101, 102, 103, 104],
    "product_id": [1, 3, 2, 5],
    "qty"       : [2, 1, 5, 3],
})

products = cudf.DataFrame({
    "product_id" : [1, 2, 3, 4, 5],
    "name"       : ["Widget", "Gadget", "Doohickey", "Thing", "Whatchamacallit"],
    "price"      : [9.99, 24.99, 4.49, 14.99, 29.99],
})

# Inner join (most common)
joined = orders.merge(products, on="product_id", how="inner")
joined["total"] = joined["qty"] * joined["price"]
print("Orders with product details:")
print(joined[["order_id", "name", "qty", "price", "total"]])

# Left join
left_joined = orders.merge(products, on="product_id", how="left")
print(f"\nLeft join rows: {len(left_joined)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Sorting and Ranking
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. SORTING")
print("=" * 60)

t0 = time.perf_counter()
sorted_df = big_df.sort_values(by=["label", "x"], ascending=[True, False])
sort_ms = (time.perf_counter() - t0) * 1000
print(f"Sort {len(big_df):,} rows by 2 columns: {sort_ms:.1f} ms")
print(sorted_df.head())

# ─────────────────────────────────────────────────────────────────────────────
# 7. String Operations (cuDF has a full string library)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. STRING OPERATIONS")
print("=" * 60)

emails = cudf.Series([
    "alice@example.com",
    "BOB@COMPANY.ORG",
    "carol@example.com",
    "dave@test.net",
    "  eve@example.com  ",   # with whitespace
])

# String methods — all run on GPU
print("Lower-cased:", emails.str.lower().to_arrow().to_pylist())
print("Strip+lower :", emails.str.strip().str.lower().to_arrow().to_pylist())

# Extract domain with regex
domains = emails.str.strip().str.extract(r"@([\w.]+)")
print(f"Domains extracted:\n{domains}")

# Contains filter
example_only = emails.str.strip().str.lower()
mask = example_only.str.contains("example")
print(f"example.com count: {int(mask.sum())}")

# Replace
cleaned = emails.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
print(f"Cleaned: {cleaned.to_arrow().to_pylist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Missing Values
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. MISSING VALUES")
print("=" * 60)

messy = cudf.DataFrame({
    "a": [1.0, None, 3.0, None, 5.0],
    "b": [None, 2.0, None, 4.0, 5.0],
    "c": ["x", None, "z", None, "w"],
})

print("Original:")
print(messy)
print(f"\nNull counts:\n{messy.isnull().sum()}")

filled = messy.copy()
filled["a"] = filled["a"].fillna(filled["a"].mean())
filled["b"] = filled["b"].fillna(0.0)
filled["c"] = filled["c"].fillna("missing")
print("\nFilled:")
print(filled)

# Drop rows with any null
dropped = messy.dropna()
print(f"\nAfter dropna: {len(dropped)} rows remaining")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Interop — cuDF ↔ pandas ↔ CuPy ↔ NumPy
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("9. INTEROPERABILITY")
print("=" * 60)

# cuDF → pandas (copies data to CPU)
pandas_df = big_df.head(5).to_pandas()
print(f"cuDF → pandas type: {type(pandas_df)}")

# pandas → cuDF (copies data to GPU)
import pandas as pd
pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
gdf = cudf.from_pandas(pdf)
print(f"pandas → cuDF type: {type(gdf)}")

# cuDF column → CuPy array (zero-copy)
col_gpu = big_df["x"].values             # returns CuPy array
print(f"cuDF col → CuPy: {type(col_gpu)}  shape={col_gpu.shape}")
mean_gpu = float(col_gpu.mean())
print(f"CuPy mean on column: {mean_gpu:.4f}")

# CuPy → cuDF Series (zero-copy)
gpu_arr = cp.arange(5, dtype=cp.float32)
s = cudf.Series(gpu_arr)
print(f"CuPy → cuDF Series: {s.values}")

print("\n✓  02_cudf.py complete")
