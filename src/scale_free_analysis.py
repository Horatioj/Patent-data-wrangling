"""
Scale-free analysis of the green-patent citation network, 1985–2025.

Treats the DOCDB-family citation graph as a complex network and tests the
classic "scale-free" hypothesis: that the in-degree (forward-citation)
distribution has a heavy, power-law tail  P(k) ~ k^{-alpha}.

Reads the pre-computed degree table written by src/network_analysis.py
(net_5yr_degree.parquet: docdb_family_id, in_degree, out_degree, category),
so it does not rebuild the ~20M-edge graph.

Figures (Clauset, Shalizi & Newman 2009 methodology):
  SF1 — In-degree CCDF (log-log) with MLE power-law fit (k_min via KS) and a
        lognormal comparison.
  SF2 — In-degree CCDF by patent category H / G / N (heavier tail = more
        influential citation hubs).
  SF3 — Citation rank-size (Zipf) plot, log-log.
  SF4 — In- vs out-degree CCDF (citations received vs references made).

Run from the project root:
    python src/scale_free_analysis.py
"""

import os
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

OUT = "PATSTAT2025FALL/output/vis"
DATA = "PATSTAT2025FALL/output"
os.makedirs(OUT, exist_ok=True)

CAT_COLOR = {"H": "#e76f51", "G": "#2a9d8f", "N": "#264653"}
CAT_LABEL = {"H": "High-influence green (H)", "G": "Green (G)", "N": "Non-green neighbor (N)"}

# Candidate lower cutoffs for the power-law tail (k_min search grid).
KMIN_GRID = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 75, 100]


# ── Power-law tools (continuous approximation with the k_min−0.5 correction) ──
def fit_powerlaw(x: np.ndarray, kmin_grid=KMIN_GRID) -> dict:
    """MLE power-law exponent with k_min chosen by KS-distance minimisation.

    For each candidate k_min, alpha_hat = 1 + n / Σ ln(x_i / (k_min − 0.5)) over
    x_i ≥ k_min; pick the k_min minimising the Kolmogorov–Smirnov distance
    between the empirical and fitted complementary CDFs.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    best = None
    for kmin in kmin_grid:
        tail = np.sort(x[x >= kmin])
        n = tail.size
        if n < 100:
            continue
        alpha = 1.0 + n / np.sum(np.log(tail / (kmin - 0.5)))
        # Empirical CDF vs continuous power-law CDF  F(x)=1−(x/kmin)^(1−alpha)
        cdf_emp = np.arange(1, n + 1) / n
        cdf_th = 1.0 - (tail / kmin) ** (1.0 - alpha)
        ks = np.max(np.abs(cdf_emp - cdf_th))
        if best is None or ks < best["ks"]:
            best = {"alpha": alpha, "kmin": kmin, "ks": ks, "n_tail": n}
    return best


def ccdf(x: np.ndarray):
    """Empirical complementary CDF  P(X ≥ k)  over the distinct positive values."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    vals = np.unique(x)
    n = x.size
    ge = n - np.searchsorted(x, vals, side="left")   # count of points ≥ val
    return vals, ge / n


# ── Load degrees ──────────────────────────────────────────────────────────────
print("Loading net_5yr_degree.parquet …")
deg = pl.read_parquet(f"{DATA}/net_5yr_degree.parquet",
                      columns=["in_degree", "out_degree", "category"])
in_deg = deg["in_degree"].to_numpy()
out_deg = deg["out_degree"].to_numpy()
cat = deg["category"].to_numpy()
print(f"  Nodes: {len(in_deg):,} · cited at least once: {(in_deg > 0).mean():.1%}\n")


# ════════════════════════════════════════════════════════════════════════════
# SF1 — In-degree CCDF with power-law + lognormal fits
# ════════════════════════════════════════════════════════════════════════════
fit = fit_powerlaw(in_deg)
vals, cc = ccdf(in_deg)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(vals, cc, s=10, color="#4a90d9", alpha=0.55, edgecolors="none",
           label="Empirical CCDF (forward citations)", zorder=3)

# Power-law fit: anchor the fitted CCDF at P(X ≥ k_min)
kmin, alpha = fit["kmin"], fit["alpha"]
anchor = cc[np.searchsorted(vals, kmin)]
xs = np.logspace(np.log10(kmin), np.log10(vals.max()), 100)
pl_ccdf = anchor * (xs / kmin) ** (1.0 - alpha)
ax.plot(xs, pl_ccdf, color="#e63946", lw=2.2, ls="--", zorder=5,
        label=fr"Power-law fit: $\alpha={alpha:.2f}$, $k_{{\min}}={kmin}$")

# Lognormal comparison fitted on the same tail
tail = in_deg[in_deg >= kmin].astype(float)
mu, sigma = np.log(tail).mean(), np.log(tail).std()
ln_ccdf = anchor * stats.norm.sf((np.log(xs) - mu) / sigma) / stats.norm.sf((np.log(kmin) - mu) / sigma)
ax.plot(xs, ln_ccdf, color="#6a4c93", lw=1.8, ls=":", zorder=4,
        label=fr"Lognormal fit ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)")

ax.axvline(kmin, color="#bbb", lw=0.8, ls="-")
ax.text(kmin * 1.05, 1e-5, fr"$k_{{\min}}={kmin}$", fontsize=8, color="#777")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Forward citations  $k$  (in-degree)", fontsize=10)
ax.set_ylabel(r"$P(X \geq k)$", fontsize=10)
ax.legend(fontsize=9, loc="lower left")
ax.set_title("Scale-Free Structure of the Green-Patent Citation Network\n"
             f"Heavy-tailed in-degree distribution (KS distance = {fit['ks']:.3f}, "
             f"n$_{{tail}}$ = {fit['n_tail']:,})", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/SF1_degree_ccdf.png")
plt.close()
print(f"Saved SF1_degree_ccdf.png  (alpha={alpha:.3f}, kmin={kmin}, KS={fit['ks']:.3f})")


# ════════════════════════════════════════════════════════════════════════════
# SF2 — In-degree CCDF by patent category
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
for c in ["N", "G", "H"]:
    sub = in_deg[cat == c]
    v, cc_c = ccdf(sub)
    lab = CAT_LABEL[c]
    # Only quote a power-law exponent where a genuine heavy tail exists. The G
    # category is citation-truncated by construction (highly-cited green
    # families are reclassified as H), so a power-law fit is not meaningful.
    if sub.max() >= 50:
        fc = fit_powerlaw(sub)
        if fc:
            lab += fr"  ($\alpha={fc['alpha']:.2f}$)"
    else:
        lab += "  (citation-truncated)"
    ax.plot(v, cc_c, color=CAT_COLOR[c], lw=2.0, alpha=0.9, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Forward citations  $k$  (in-degree)", fontsize=10)
ax.set_ylabel(r"$P(X \geq k)$", fontsize=10)
ax.legend(fontsize=9, loc="lower left", title="Category (power-law exponent)",
          title_fontsize=9)
ax.set_title("Citation In-Degree Tails by Patent Category\n"
             "Flatter tail = more highly-cited hubs (H patents are the heaviest-tailed)",
             fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/SF2_ccdf_by_category.png")
plt.close()
print("Saved SF2_ccdf_by_category.png")


# ════════════════════════════════════════════════════════════════════════════
# SF3 — Citation rank-size (Zipf) plot
# ════════════════════════════════════════════════════════════════════════════
sorted_deg = np.sort(in_deg[in_deg > 0])[::-1]
ranks = np.arange(1, sorted_deg.size + 1)
# Thin to keep the PNG light while preserving the curve shape
if sorted_deg.size > 60000:
    idx = np.unique(np.round(np.logspace(0, np.log10(sorted_deg.size - 1), 60000)).astype(int))
    ranks, sorted_deg = ranks[idx], sorted_deg[idx]

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.scatter(ranks, sorted_deg, s=5, color="#2a9d8f", alpha=0.5, edgecolors="none")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Rank (most-cited family = 1)", fontsize=10)
ax.set_ylabel("Forward citations", fontsize=10)
ax.set_title("Citation Rank-Size (Zipf) Distribution — Green Patent Families\n"
             "A near-linear log-log decline is the signature of a scale-free network",
             fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/SF3_zipf_rank_size.png")
plt.close()
print("Saved SF3_zipf_rank_size.png")


# ════════════════════════════════════════════════════════════════════════════
# SF4 — In- vs out-degree CCDF (citations received vs references made)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6.5))
for arr, color, label in [(in_deg, "#e63946", "In-degree (citations received)"),
                          (out_deg, "#457b9d", "Out-degree (references made)")]:
    v, cc_a = ccdf(arr)
    ax.plot(v, cc_a, color=color, lw=2.0, label=label)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Degree  $k$", fontsize=10)
ax.set_ylabel(r"$P(X \geq k)$", fontsize=10)
ax.legend(fontsize=9, loc="lower left")
ax.set_title("In- vs Out-Degree Distributions of the Citation Network\n"
             "Citations received are far more heavy-tailed than references made",
             fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/SF4_in_out_degree_ccdf.png")
plt.close()
print("Saved SF4_in_out_degree_ccdf.png")

print(f"\nAll scale-free figures saved to: {OUT}/")
