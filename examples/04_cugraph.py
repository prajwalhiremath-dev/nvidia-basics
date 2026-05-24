"""
04_cugraph.py — cuGraph: GPU Graph Analytics

cuGraph is GPU-accelerated NetworkX.  It reads cuDF edge-list DataFrames
natively — the graph never leaves the GPU.

Algorithms covered:
  PageRank              — node importance (like Google's original algorithm)
  BFS                   — Breadth-First Search (shortest hops)
  SSSP                  — Single-Source Shortest Paths (weighted)
  Connected Components  — find isolated subgraphs
  Louvain               — community detection (modularity maximisation)
  Triangle Count        — counts closed triangles per node
  Jaccard Similarity    — link prediction / edge similarity
  Degree Centrality     — simple measure of how connected a node is
  Betweenness Centrality— how often a node is on a shortest path
"""

import time
import numpy as np
import cudf
import cugraph

print(f"cuGraph version: {cugraph.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def timed(label, fn):
    t0 = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label}: {ms:.1f} ms")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Build synthetic graphs
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

# --- Small graph (visible, easy to inspect) ---
# 8-node graph:  0-1-2-3 (chain) + triangle 4-5-6 + bridge 3-4 + isolated 7
small_src = [0, 1, 2, 3, 4, 5, 5, 3]
small_dst = [1, 2, 3, 4, 5, 6, 4, 7]
small_wgt = [1.0, 2.0, 1.5, 3.0, 1.0, 2.5, 1.0, 0.5]

small_edges = cudf.DataFrame({
    "src": cudf.Series(small_src, dtype="int32"),
    "dst": cudf.Series(small_dst, dtype="int32"),
    "wgt": cudf.Series(small_wgt, dtype="float32"),
})

# --- Large random graph (scale test) ---
N_nodes = 50_000
N_edges = 500_000
src_large = rng.integers(0, N_nodes, N_edges).astype(np.int32)
dst_large = rng.integers(0, N_nodes, N_edges).astype(np.int32)
wgt_large = rng.uniform(0.1, 10.0, N_edges).astype(np.float32)

large_edges = cudf.DataFrame({
    "src": cudf.Series(src_large),
    "dst": cudf.Series(dst_large),
    "wgt": cudf.Series(wgt_large),
})

print(f"Small graph : {len(set(small_src+small_dst))} nodes, {len(small_src)} edges")
print(f"Large graph : {N_nodes:,} nodes, {N_edges:,} edges")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build cuGraph Graph objects
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("1. GRAPH CREATION")
print("=" * 60)

# Undirected graph (no direction on edges)
G_small = cugraph.Graph()
G_small.from_cudf_edgelist(
    small_edges, source="src", destination="dst", edge_attr="wgt"
)

# Directed graph (edges have direction, e.g., web links)
DG_small = cugraph.MultiDiGraph()
DG_small.from_cudf_edgelist(
    small_edges, source="src", destination="dst", edge_attr="wgt"
)

# Large undirected graph
G_large = cugraph.Graph()
timed("Build large graph (50k nodes, 500k edges)",
      lambda: G_large.from_cudf_edgelist(
          large_edges, source="src", destination="dst", edge_attr="wgt"
      ))

print(f"  G_small nodes: {G_small.number_of_nodes()}  edges: {G_small.number_of_edges()}")
print(f"  G_large nodes: {G_large.number_of_nodes():,}  edges: {G_large.number_of_edges():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PageRank — who are the most important nodes?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. PAGERANK")
print("=" * 60)

# Small graph — manual inspection
pr_small = cugraph.pagerank(G_small, alpha=0.85, max_iter=100, tol=1e-5)
pr_small = pr_small.sort_values("pagerank", ascending=False).reset_index(drop=True)
print("PageRank scores (small graph, sorted):")
print(pr_small)

# Large graph — performance test
pr_large = timed("PageRank (50k nodes)",
                 lambda: cugraph.pagerank(G_large, alpha=0.85, max_iter=100))
top5 = pr_large.sort_values("pagerank", ascending=False).head()
print(f"\n  Top-5 PageRank nodes:\n{top5}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BFS — Breadth-First Search
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. BFS — BREADTH-FIRST SEARCH")
print("=" * 60)

# BFS from source node 0 — finds shortest hops to every reachable node
bfs_small = cugraph.bfs(G_small, start=0)
print("BFS from node 0 (small graph):")
print(bfs_small.sort_values("vertex"))
# Columns: vertex, distance (hops), predecessor

# Large graph BFS
bfs_large = timed("BFS (50k nodes, source=0)",
                  lambda: cugraph.bfs(G_large, start=0))
reachable = int((bfs_large["distance"] < 2**31 - 1).sum())
print(f"\n  Nodes reachable from 0: {reachable:,} / {N_nodes:,}")
avg_dist = float(bfs_large[bfs_large["distance"] < 2**31-1]["distance"].mean())
print(f"  Average BFS distance  : {avg_dist:.2f} hops")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SSSP — Single-Source Shortest Paths (weighted)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. SSSP — WEIGHTED SHORTEST PATHS")
print("=" * 60)

sssp_small = cugraph.sssp(G_small, source=0)
print("Shortest path distances from node 0 (small, weighted):")
print(sssp_small.sort_values("vertex"))
# Columns: vertex, distance (sum of edge weights), predecessor

# Large graph
sssp_large = timed("SSSP (50k nodes, source=0)",
                   lambda: cugraph.sssp(G_large, source=0))
finite_mask = sssp_large["distance"] < 1e38
avg_wt = float(sssp_large[finite_mask]["distance"].mean())
print(f"\n  Average weighted distance: {avg_wt:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Connected Components
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. CONNECTED COMPONENTS")
print("=" * 60)

# Weakly connected components (ignores edge direction)
wcc = cugraph.weakly_connected_components(G_small)
print("Weakly connected components (small graph):")
print(wcc.sort_values("vertex"))
n_components = len(wcc["labels"].unique())
print(f"  Number of components: {n_components}")
# Node 7 should be in its own component since it's only reached via bridge 3→7

# Large graph
wcc_large = timed("Weakly CC (50k nodes)",
                  lambda: cugraph.weakly_connected_components(G_large))
n_comp_large = len(wcc_large["labels"].unique())
print(f"\n  Large graph components: {n_comp_large:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Louvain — Community Detection
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. LOUVAIN — COMMUNITY DETECTION")
print("=" * 60)

# Louvain finds groups of nodes that are more densely connected
# among themselves than to the rest of the graph.
# Returns (partition DataFrame, modularity score).

parts_small, modularity_small = cugraph.louvain(G_small, max_iter=100)
print("Communities in small graph:")
print(parts_small.sort_values("vertex"))
print(f"  Modularity score: {modularity_small:.4f}  (higher = better separation)")

# Large graph
result_large = timed(
    "Louvain (50k nodes)",
    lambda: cugraph.louvain(G_large, max_iter=100)
)
parts_large, mod_large = result_large
n_communities = len(parts_large["partition"].unique())
print(f"\n  Large graph communities found: {n_communities:,}")
print(f"  Modularity: {mod_large:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Triangle Count — how "clustered" is each node?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. TRIANGLE COUNT")
print("=" * 60)

# A triangle = 3 nodes all connected to each other.
# Nodes in dense cliques have high triangle counts.
tc_small = cugraph.triangle_count(G_small)
print("Triangle counts (small graph):")
print(tc_small.sort_values("vertex"))
# Nodes 4,5,6 form a triangle, so they should have count > 0

# Large graph
tc_large = timed("Triangle Count (50k nodes)",
                 lambda: cugraph.triangle_count(G_large))
total_triangles = int(tc_large["counts"].sum()) // 3
print(f"\n  Total triangles in large graph: {total_triangles:,}")
avg_tc = float(tc_large["counts"].mean())
print(f"  Average per node: {avg_tc:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Jaccard Similarity — link prediction
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. JACCARD SIMILARITY")
print("=" * 60)

# Jaccard similarity between two nodes = |neighbors(u) ∩ neighbors(v)| /
#                                         |neighbors(u) ∪ neighbors(v)|
# Used to predict which new edges are likely to form.
jac = cugraph.jaccard(G_small)
print("Jaccard similarity (small graph, all pairs):")
print(jac.sort_values("jaccard_coeff", ascending=False).head(10))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Degree Centrality
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("9. DEGREE CENTRALITY")
print("=" * 60)

# Degree centrality = number of connections / (N-1).
# The simplest centrality measure — high degree = well-connected hub.
dc = cugraph.degree_centrality(G_small)
print("Degree centrality (small graph):")
print(dc.sort_values("degree_centrality", ascending=False))

dc_large = timed("Degree Centrality (50k nodes)",
                 lambda: cugraph.degree_centrality(G_large))
top_hubs = dc_large.sort_values("degree_centrality", ascending=False).head()
print(f"\n  Top-5 hubs:\n{top_hubs}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. End-to-end: cuDF → cuGraph pipeline
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("10. END-TO-END: cuDF PIPELINE → cuGraph ANALYSIS")
print("=" * 60)

# Simulate a social-network dataset: users interacting with posts
n_users = 1000
n_inter = 5000
user_ids  = rng.integers(0, n_users, n_inter).astype(np.int32)
post_ids  = rng.integers(n_users, n_users + 500, n_inter).astype(np.int32)
strength  = rng.uniform(1, 10, n_inter).astype(np.float32)

# Step 1: Load + clean with cuDF
raw = cudf.DataFrame({"user": user_ids, "post": post_ids, "strength": strength})
# Aggregate duplicate (user, post) pairs
agg = raw.groupby(["user", "post"])["strength"].sum().reset_index()
agg.columns = cudf.Index(["src", "dst", "wgt"])
print(f"Raw interactions: {len(raw):,}   After dedup: {len(agg):,}")

# Step 2: Build graph
G_social = cugraph.Graph()
G_social.from_cudf_edgelist(agg, source="src", destination="dst", edge_attr="wgt")

# Step 3: PageRank to find influential users
pr = cugraph.pagerank(G_social)
top_users = pr[pr["vertex"] < n_users].sort_values(
    "pagerank", ascending=False
).head()
print(f"Most influential users (by PageRank):\n{top_users}")

# Step 4: Community detection
parts, mod = cugraph.louvain(G_social)
user_communities = parts[parts["vertex"] < n_users]
n_comm = len(user_communities["partition"].unique())
print(f"User communities found: {n_comm}  (modularity={mod:.4f})")

print("\n✓  04_cugraph.py complete")
