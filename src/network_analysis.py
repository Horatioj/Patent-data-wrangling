"""
Patent citation network analysis (DOCDB family level).

Memory-efficient: uses scipy sparse matrices (~12 bytes/edge) instead of
NetworkX (~200 bytes/edge). A 20M-edge graph uses ~240MB vs 4GB+.

Supports optional 5-year citation window to control time bias
(older patents accumulate more citations mechanically).

Input:
  citation_edges_categorized.parquet  (from find_neighbor.py)
  docdb_family_year.parquet           (from high_influence.py)

Categories: H (high-influential green), G (green), N (non-green neighbor)
"""

import polars as pl
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc

# ============================================================================
# Configuration
# ============================================================================
USE_5YR_WINDOW = True           # set False for full citation network
CITATION_WINDOW_YEARS = 5
OUTPUT_PREFIX = "net_5yr" if USE_5YR_WINDOW else "net_full"

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.figsize": (8, 5),
})

# ============================================================================
# Helper functions
# ============================================================================

def build_node_index(edges: pl.DataFrame) -> tuple[dict[int, int], np.ndarray]:
    """Map family IDs to contiguous 0-based indices. Returns (id→idx, idx→id)."""
    all_ids = pl.concat([
        edges.select(pl.col("docdb_family_id").alias("fid")),
        edges.select(pl.col("cited_docdb_family_id").alias("fid")),
    ])["fid"].unique().sort().to_numpy()
    id_to_idx = {int(fid): i for i, fid in enumerate(all_ids)}
    return id_to_idx, all_ids


def edges_to_sparse(edges: pl.DataFrame, id_to_idx: dict, n: int) -> sparse.csr_matrix:
    """Build CSR adjacency matrix (citing→cited) from edge DataFrame."""
    rows = np.array([id_to_idx[x] for x in edges["docdb_family_id"].to_list()], dtype=np.int32)
    cols = np.array([id_to_idx[x] for x in edges["cited_docdb_family_id"].to_list()], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def compute_degrees(adj: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return (out_degree, in_degree) arrays."""
    out_deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.int32)
    in_deg = np.asarray(adj.sum(axis=0)).ravel().astype(np.int32)
    return out_deg, in_deg


def pagerank_sparse(adj: sparse.csr_matrix, alpha: float = 0.85,
                    max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Power-iteration PageRank on sparse matrix. ~10x less RAM than NetworkX."""
    n = adj.shape[0]
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    out_deg[out_deg == 0] = 1  # dangling nodes
    D_inv = sparse.diags(1.0 / out_deg)
    M = (D_inv @ adj).T  # column-stochastic transition matrix

    pr = np.ones(n, dtype=np.float64) / n
    for _ in range(max_iter):
        pr_new = alpha * M.dot(pr) + (1 - alpha) / n
        pr_new /= pr_new.sum()
        if np.abs(pr_new - pr).max() < tol:
            break
        pr = pr_new
    return pr


def hits_sparse(adj: sparse.csr_matrix,
                max_iter: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """HITS hub & authority scores via power iteration on sparse matrix."""
    n = adj.shape[0]
    At = adj.T
    a = np.ones(n, dtype=np.float64) / np.sqrt(n)  # authority
    for _ in range(max_iter):
        h = adj.dot(a)
        h /= (np.linalg.norm(h) + 1e-15)
        a_new = At.dot(h)
        a_new /= (np.linalg.norm(a_new) + 1e-15)
        if np.abs(a_new - a).max() < tol:
            break
        a = a_new
    return h, a


def fit_power_law_mle(degrees: np.ndarray, k_min: int = 1) -> dict:
    """MLE estimate of power-law exponent alpha for P(k) ~ k^{-alpha}, k >= k_min."""
    tail = degrees[degrees >= k_min]
    n = len(tail)
    if n < 30:
        return {"alpha": None, "k_min": k_min, "n_tail": n}
    alpha = 1.0 + n / np.log(tail / (k_min - 0.5)).sum()
    return {"alpha": round(float(alpha), 4), "k_min": k_min, "n_tail": n}


def degree_stats_by_category(degree_df: pl.DataFrame, categories: list[str]) -> pl.DataFrame:
    rows = []
    for cat in categories:
        sub = degree_df.filter(pl.col("category") == cat)
        if sub.height == 0:
            continue
        for metric in ["in_degree", "out_degree", "total_degree"]:
            v = sub[metric]
            rows.append({
                "category": cat, "metric": metric, "count": sub.height,
                "mean": round(float(v.mean()), 4),
                "median": round(float(v.median()), 4),
                "std": round(float(v.std()), 4),
                "max": int(v.max()),
                "p90": round(float(v.quantile(0.90)), 4),
                "p95": round(float(v.quantile(0.95)), 4),
                "p99": round(float(v.quantile(0.99)), 4),
            })
    return pl.DataFrame(rows)


# ============================================================================
# Plotting functions
# ============================================================================

def plot_degree_distribution(in_deg: np.ndarray, out_deg: np.ndarray,
                             prefix: str = OUTPUT_PREFIX):
    """Log-log degree distribution with power-law reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, deg, label in [(axes[0], in_deg, "In-degree"), (axes[1], out_deg, "Out-degree")]:
        vals, counts = np.unique(deg[deg > 0], return_counts=True)
        ax.scatter(vals, counts, s=6, alpha=0.5, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label} distribution (log-log)")

        result = fit_power_law_mle(deg[deg > 0], k_min=2)
        if result["alpha"]:
            x_ref = np.logspace(np.log10(2), np.log10(vals.max()), 50)
            y_ref = x_ref ** (-result["alpha"])
            y_ref = y_ref * (counts.max() / y_ref[0]) * 0.5
            ax.plot(x_ref, y_ref, "r--", lw=1,
                    label=fr"$\alpha={result['alpha']:.2f}$ (k$\geq$2)")
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"PATSTAT2025FALL/output/{prefix}_degree_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved {prefix}_degree_distribution.png")


def plot_degree_by_category(degree_df: pl.DataFrame, prefix: str = OUTPUT_PREFIX):
    """Box plot of in-degree by category."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["H", "G", "N"]
    data = []
    labels = []
    for cat in cats:
        vals = degree_df.filter(pl.col("category") == cat)["in_degree"].to_numpy()
        if len(vals) > 0:
            data.append(vals)
            labels.append(f"{cat} (n={len(vals):,})")

    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("In-degree (forward citations)")
    ax.set_title("In-degree distribution by patent category")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(f"PATSTAT2025FALL/output/{prefix}_degree_by_category.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved {prefix}_degree_by_category.png")


def plot_pagerank_by_category(pr_df: pl.DataFrame, prefix: str = OUTPUT_PREFIX):
    """Violin plot of PageRank by category."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["H", "G", "N"]
    data = []
    labels = []
    for cat in cats:
        vals = pr_df.filter(pl.col("category") == cat)["pagerank"].to_numpy()
        if len(vals) > 0:
            data.append(vals)
            labels.append(f"{cat} (n={len(vals):,})")

    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("PageRank")
    ax.set_title("PageRank distribution by patent category")
    plt.tight_layout()
    plt.savefig(f"PATSTAT2025FALL/output/{prefix}_pagerank_by_category.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved {prefix}_pagerank_by_category.png")


def plot_mixing_heatmap(df_mixing: pl.DataFrame, prefix: str = OUTPUT_PREFIX):
    """Heatmap of citation flow between categories."""
    pivot = df_mixing.to_pandas().pivot_table(
        index="from_category", columns="to_category",
        values="fraction", fill_value=0,
    )
    cats_order = [c for c in ["H", "G", "N", "O"] if c in pivot.index]
    pivot = pivot.reindex(index=cats_order, columns=cats_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cats_order)))
    ax.set_xticklabels(cats_order)
    ax.set_yticks(range(len(cats_order)))
    ax.set_yticklabels(cats_order)
    ax.set_xlabel("Cited category")
    ax.set_ylabel("Citing category")
    ax.set_title("Citation flow between categories (fraction of edges)")

    for i in range(len(cats_order)):
        for j in range(len(cats_order)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.15 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"PATSTAT2025FALL/output/{prefix}_mixing_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved {prefix}_mixing_heatmap.png")


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    # --- Load and optionally window edges ---
    print("Loading citation edges...")
    edges = pl.read_parquet("PATSTAT2025FALL/output/citation_edges_categorized.parquet")

    if USE_5YR_WINDOW:
        print(f"Applying {CITATION_WINDOW_YEARS}-year citation window...")
        family_years = pl.read_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet")

        edges = (
            edges
            .join(
                family_years.rename({"family_year": "citing_year"}),
                on="docdb_family_id", how="left",
            )
            .join(
                family_years.rename({
                    "docdb_family_id": "cited_docdb_family_id",
                    "family_year": "cited_year",
                }),
                on="cited_docdb_family_id", how="left",
            )
            .filter(
                pl.col("citing_year").is_not_null()
                & pl.col("cited_year").is_not_null()
                & (pl.col("citing_year") >= pl.col("cited_year"))
                & (pl.col("citing_year") <= pl.col("cited_year") + CITATION_WINDOW_YEARS)
            )
            .drop(["citing_year", "cited_year"])
        )
        del family_years
        gc.collect()

    print(f"Edges after filtering: {edges.height:,}")

    # --- Build sparse graph ---
    print("Building node index and sparse matrix...")
    id_to_idx, idx_to_id = build_node_index(edges)
    n = len(id_to_idx)

    # Build category array aligned with indices
    cats_citing = edges.select([
        pl.col("docdb_family_id").alias("fid"),
        pl.col("citing_category").alias("cat"),
    ])
    cats_cited = edges.select([
        pl.col("cited_docdb_family_id").alias("fid"),
        pl.col("cited_category").alias("cat"),
    ])
    cat_df = pl.concat([cats_citing, cats_cited]).unique(subset=["fid"])
    cat_map = dict(zip(cat_df["fid"].to_list(), cat_df["cat"].to_list()))
    cat_array = np.array([cat_map.get(int(fid), "O") for fid in idx_to_id])
    del cats_citing, cats_cited, cat_df

    adj = edges_to_sparse(edges, id_to_idx, n)
    del edges
    gc.collect()

    print(f"Graph: {n:,} nodes, {adj.nnz:,} edges, "
          f"sparse matrix: {adj.data.nbytes / 1e6:.1f} MB")

    # --- Degree analysis (pure numpy, no graph object) ---
    print("\nComputing degrees...")
    out_deg, in_deg = compute_degrees(adj)

    degree_df = pl.DataFrame({
        "docdb_family_id": idx_to_id.tolist(),
        "in_degree": in_deg.tolist(),
        "out_degree": out_deg.tolist(),
    }).with_columns([
        (pl.col("in_degree") + pl.col("out_degree")).alias("total_degree"),
        pl.Series("category", cat_array.tolist()),
    ])
    degree_df.write_parquet(f"PATSTAT2025FALL/output/{OUTPUT_PREFIX}_degree.parquet", compression="zstd")

    df_degree_stats = degree_stats_by_category(degree_df, ["H", "G", "N"])
    print(df_degree_stats)

    # --- Power-law diagnostics ---
    print("\nPower-law fitting (in-degree)...")
    non_zero_in = in_deg[in_deg > 0]

    powerlaw_rows = []
    for k_min in [1, 2, 5, 10]:
        r = fit_power_law_mle(non_zero_in, k_min)
        r["scope"] = "all nodes"
        powerlaw_rows.append(r)
        print(f"  k_min={k_min}: alpha={r['alpha']}, n_tail={r['n_tail']}")

    for cat in ["H", "G", "N"]:
        cat_in = in_deg[cat_array == cat]
        cat_nz = cat_in[cat_in > 0]
        r = fit_power_law_mle(cat_nz, k_min=1)
        r["scope"] = f"category {cat}"
        powerlaw_rows.append(r)
        print(f"  {cat}: alpha={r['alpha']}, n_tail={r['n_tail']}")

    df_powerlaw = pl.DataFrame(powerlaw_rows)

    # In-degree frequency table
    vals, counts = np.unique(in_deg, return_counts=True)
    df_in_dist = pl.DataFrame({"k": vals.tolist(), "count": counts.tolist()})
    df_in_dist.write_parquet(f"PATSTAT2025FALL/output/{OUTPUT_PREFIX}_in_degree_dist.parquet", compression="zstd")

    # --- PageRank (sparse, ~10x cheaper than NetworkX) ---
    print("\nComputing PageRank...")
    pr_vals = pagerank_sparse(adj)

    pr_df = pl.DataFrame({
        "docdb_family_id": idx_to_id.tolist(),
        "pagerank": pr_vals.tolist(),
    }).with_columns(pl.Series("category", cat_array.tolist()))
    pr_df.write_parquet(f"PATSTAT2025FALL/output/{OUTPUT_PREFIX}_pagerank.parquet", compression="zstd")

    pr_stats_rows = []
    for cat in ["H", "G", "N"]:
        sub = pr_df.filter(pl.col("category") == cat)
        if sub.height == 0:
            continue
        v = sub["pagerank"]
        pr_stats_rows.append({
            "category": cat, "n_nodes": sub.height,
            "mean": round(float(v.mean()), 8),
            "median": round(float(v.median()), 8),
            "max": round(float(v.max()), 8),
            "p99": round(float(v.quantile(0.99)), 8),
            "sum": round(float(v.sum()), 6),
        })
    df_pr_stats = pl.DataFrame(pr_stats_rows)
    print(df_pr_stats)

    top_pr = pr_df.sort("pagerank", descending=True).head(50)

    # --- HITS (sparse) ---
    print("\nComputing HITS...")
    hub_vals, auth_vals = hits_sparse(adj)

    hits_df = pl.DataFrame({
        "docdb_family_id": idx_to_id.tolist(),
        "hub_score": hub_vals.tolist(),
        "authority_score": auth_vals.tolist(),
    }).with_columns(pl.Series("category", cat_array.tolist()))
    hits_df.write_parquet(f"PATSTAT2025FALL/output/{OUTPUT_PREFIX}_hits.parquet", compression="zstd")

    hits_rows = []
    for cat in ["H", "G", "N"]:
        sub = hits_df.filter(pl.col("category") == cat)
        if sub.height == 0:
            continue
        hits_rows.append({
            "category": cat,
            "mean_hub": round(float(sub["hub_score"].mean()), 8),
            "max_hub": round(float(sub["hub_score"].max()), 8),
            "mean_authority": round(float(sub["authority_score"].mean()), 8),
            "max_authority": round(float(sub["authority_score"].max()), 8),
        })
    df_hits_stats = pl.DataFrame(hits_rows)
    print(df_hits_stats)

    # --- Mixing matrix (computed from sparse matrix + category arrays, no reload) ---
    print("\nComputing category mixing from sparse matrix...")
    coo = adj.tocoo()
    src_cats = cat_array[coo.row]
    dst_cats = cat_array[coo.col]
    mixing_keys, mixing_counts = np.unique(
        np.column_stack([src_cats, dst_cats]), axis=0, return_counts=True
    )
    total_e = int(mixing_counts.sum())
    df_mixing = pl.DataFrame({
        "from_category": mixing_keys[:, 0].tolist(),
        "to_category": mixing_keys[:, 1].tolist(),
        "n_edges": mixing_counts.tolist(),
    }).with_columns(
        (pl.col("n_edges") / total_e).round(6).alias("fraction")
    ).sort(["from_category", "to_category"])
    del coo, src_cats, dst_cats

    # --- Degree assortativity (Pearson r on edge endpoint degrees) ---
    # Sample edges from the sparse matrix to keep memory bounded
    print("  Computing degree assortativity from sparse matrix...")
    coo = adj.tocoo()
    n_sample = min(len(coo.row), 2_000_000)
    if n_sample > 0:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(len(coo.row), size=n_sample, replace=False)
        src_total = out_deg[coo.row[idx_sample]] + in_deg[coo.row[idx_sample]]
        dst_total = out_deg[coo.col[idx_sample]] + in_deg[coo.col[idx_sample]]
        degree_assortativity = float(np.corrcoef(src_total, dst_total)[0, 1])
    else:
        degree_assortativity = 0.0
    del coo
    print(f"  Degree assortativity (sampled): {degree_assortativity:.6f}")

    # --- Connected components (scipy, much cheaper than NetworkX) ---
    print("\nComputing connected components...")
    adj_sym = adj + adj.T  # undirected
    n_components, labels = connected_components(adj_sym, directed=False)
    comp_sizes = np.bincount(labels)
    largest_cc_idx = comp_sizes.argmax()
    largest_cc_size = int(comp_sizes[largest_cc_idx])

    n_singletons = int((comp_sizes == 1).sum())
    print(f"  Components: {n_components:,}, largest: {largest_cc_size:,} "
          f"({largest_cc_size/n:.1%} of nodes), singletons: {n_singletons:,}")

    # Category composition of largest CC
    largest_mask = labels == largest_cc_idx
    cc_cats = cat_array[largest_mask]
    cc_unique, cc_counts = np.unique(cc_cats, return_counts=True)
    df_cc = pl.DataFrame({
        "category": cc_unique.tolist(),
        "n_nodes": cc_counts.tolist(),
        "fraction_of_cc": (cc_counts / largest_cc_size).round(4).tolist(),
    })

    del adj_sym
    gc.collect()

    # --- Plots ---
    print("\nGenerating plots...")
    plot_degree_distribution(in_deg, out_deg)
    plot_degree_by_category(degree_df)
    plot_pagerank_by_category(pr_df)
    plot_mixing_heatmap(df_mixing)

    # --- Save Excel ---
    print("\nSaving to Excel...")
    df_summary = pl.DataFrame([
        {"Metric": "Nodes", "Value": f"{n:,}"},
        {"Metric": "Edges", "Value": f"{adj.nnz:,}"},
        {"Metric": "Density", "Value": f"{adj.nnz / (n * (n-1)):.2e}"},
        {"Metric": "Citation window", "Value": f"{CITATION_WINDOW_YEARS}yr" if USE_5YR_WINDOW else "full"},
        {"Metric": "Degree assortativity (approx)", "Value": f"{degree_assortativity:.6f}"},
        {"Metric": "Connected components", "Value": f"{n_components:,}"},
        {"Metric": "Largest CC size", "Value": f"{largest_cc_size:,}"},
        {"Metric": "Largest CC fraction", "Value": f"{largest_cc_size/n:.4f}"},
        {"Metric": "Singletons", "Value": f"{n_singletons:,}"},
    ])

    cat_counts = degree_df.group_by("category").agg(pl.len().alias("n_nodes")).sort("category")

    with pd.ExcelWriter(f"PATSTAT2025FALL/output/{OUTPUT_PREFIX}_analysis.xlsx", engine="openpyxl") as writer:
        df_summary.to_pandas().to_excel(writer, sheet_name="Graph Summary", index=False)
        cat_counts.to_pandas().to_excel(writer, sheet_name="Nodes by Category", index=False)
        df_degree_stats.to_pandas().to_excel(writer, sheet_name="Degree Stats", index=False)
        df_powerlaw.to_pandas().to_excel(writer, sheet_name="Power-Law Fit", index=False)
        df_in_dist.to_pandas().to_excel(writer, sheet_name="In-Degree Distribution", index=False)
        df_pr_stats.to_pandas().to_excel(writer, sheet_name="PageRank Stats", index=False)
        top_pr.to_pandas().to_excel(writer, sheet_name="Top 50 PageRank", index=False)
        df_hits_stats.to_pandas().to_excel(writer, sheet_name="HITS Stats", index=False)
        df_mixing.to_pandas().to_excel(writer, sheet_name="Category Mixing", index=False)
        df_cc.to_pandas().to_excel(writer, sheet_name="Largest CC Composition", index=False)

    print(f"Saved {OUTPUT_PREFIX}_analysis.xlsx")
    print(f"\nParquet outputs: {OUTPUT_PREFIX}_degree.parquet, "
          f"{OUTPUT_PREFIX}_pagerank.parquet, {OUTPUT_PREFIX}_hits.parquet, "
          f"{OUTPUT_PREFIX}_in_degree_dist.parquet")

    del adj
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
