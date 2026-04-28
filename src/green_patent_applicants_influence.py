"""
Green patent analysis — applicants, influential patents, network science.

Sections:
  D. Applicant / inventor analysis  (companies, universities, gov, individuals)
       — harmonised names via han_id/han_name; TW excluded
  E. Most influential green patents
       — forward citations, OECD quality index, family size, PageRank
       — scientific scatter-bubble figure
  F. Network science  (Dechezleprêtre et al. 2015 approach)
       — PageRank distribution across green-technology sectors
       — HITS authority scores
       — Innovation-influence Lorenz curve
       — Sankey-style citation flow: H / G / N categories

Run from the project root:
    python src/green_patent_applicants_influence.py

References:
  Dechezleprêtre, A., Martin, R., Mohnen, M. (2014/2015).
      "Knowledge Spillovers from Clean and Dirty Technologies."
      CEP Discussion Paper No. 1300.
  Hall, B.H., Jaffe, A., Trajtenberg, M. (2001).
      "The NBER Patent Citation Data File." NBER WP 8498.
  OECD (2009). "OECD Patent Quality Indicators." STI Working Paper.
"""

import os, warnings
import numpy as np
import polars as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import scipy.stats as stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

OUT = "PATSTAT2025FALL/output/vis"
os.makedirs(OUT, exist_ok=True)

EXCLUDE_CTRY = {"TW"}   # exclude Taiwan per user request

SECTOR_COLOR = {
    "Energy":           "#2a9d8f",
    "Transportation":   "#264653",
    "Manufacturing":    "#e9c46a",
    "Buildings":        "#e76f51",
    "ICT":              "#6a4c93",
    "Waste management": "#81b29a",
    "Agriculture":      "#f4a261",
    "CCS":              "#a8dadc",
    "Mixed":            "#a8a8a8",
}
SECTORS_ORDERED = ["Energy", "Transportation", "Manufacturing", "Buildings",
                   "ICT", "Waste management", "Agriculture"]

ENTITY_COLOR = {
    "Company":        "#264653",
    "University":     "#2a9d8f",
    "Gov/Non-profit": "#e9c46a",
    "Individual":     "#e76f51",
    "Unknown/Other":  "#a8a8a8",
}


# ═══════════════════════════════════════════════════════════════════════════
# 0. Load data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading green_patent8526.parquet …")
green = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet").filter(
    ~pl.col("appln_auth").is_in(list(EXCLUDE_CTRY))
).with_columns(
    pl.col("appln_filing_date").str.slice(0, 4).cast(pl.Int32).alias("yr")
)

print("Loading patent_quality_family.parquet …")
pq = pl.read_parquet("PATSTAT2025FALL/output/patent_quality_family.parquet")

print("Loading net_5yr_pagerank.parquet …")
pr = pl.read_parquet("PATSTAT2025FALL/output/net_5yr_pagerank.parquet").with_columns(
    pl.col("docdb_family_id").cast(pl.Int32)
)

print("Loading net_5yr_hits.parquet …")
hits = pl.read_parquet("PATSTAT2025FALL/output/net_5yr_hits.parquet").with_columns(
    pl.col("docdb_family_id").cast(pl.Int32)
)

print("Loading citation_edges_categorized.parquet …")
cit = pl.read_parquet("PATSTAT2025FALL/output/citation_edges_categorized.parquet")

print("All data loaded.\n")

# ═══════════════════════════════════════════════════════════════════════════
# Helper: classify psn_sector string → clean entity type
# ═══════════════════════════════════════════════════════════════════════════
def classify_entity(psn_str: str | None) -> str:
    if not psn_str:
        return "Unknown/Other"
    s = psn_str.upper()
    if "COMPANY" in s:
        return "Company"
    if "UNIVERSITY" in s:
        return "University"
    if "GOV NON-PROFIT" in s or "GOV" in s:
        return "Gov/Non-profit"
    if "INDIVIDUAL" in s:
        return "Individual"
    return "Unknown/Other"

# Apply to green
green = green.with_columns(
    pl.col("psn_sector")
    .map_elements(classify_entity, return_dtype=pl.Utf8)
    .alias("entity_type")
)

# Primary sector per patent (first comma-delimited entry)
green = green.with_columns(
    pl.col("sector").str.split(",").list.first().str.strip_chars().alias("primary_sector")
)


# ═══════════════════════════════════════════════════════════════════════════
# D1. Entity type breakdown (stacked bar by year)
# ═══════════════════════════════════════════════════════════════════════════
YEAR_MIN, YEAR_MAX = 1985, 2025
ent_yr = (
    green
    .filter((pl.col("yr") >= YEAR_MIN) & (pl.col("yr") <= YEAR_MAX))
    .group_by(["yr", "entity_type"]).agg(pl.len().alias("n"))
    .to_pandas()
    .pivot(index="yr", columns="entity_type", values="n")
    .fillna(0).astype(int)
    .reindex(range(YEAR_MIN, YEAR_MAX + 1), fill_value=0)
)

ENTITY_ORDER = ["Company", "University", "Gov/Non-profit", "Individual", "Unknown/Other"]
ent_yr = ent_yr.reindex(columns=[c for c in ENTITY_ORDER if c in ent_yr.columns], fill_value=0)

fig, ax = plt.subplots(figsize=(13, 5.5))
bottom = np.zeros(len(ent_yr))
years = ent_yr.index.values
for etype in ENTITY_ORDER:
    if etype not in ent_yr.columns:
        continue
    vals = ent_yr[etype].values / 1000
    ax.bar(years, vals, bottom=bottom / 1000,
           label=etype, color=ENTITY_COLOR[etype], width=0.85, alpha=0.90)
    bottom += ent_yr[etype].values

ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel("Green patent applications (thousands)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
ax.legend(loc="upper left", fontsize=9, ncol=2)
plt.title(
    "Green Patent Applications by Applicant Type 1990–2023\n"
    "(Harmonised via han_id; TW excluded)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/D1_entity_type_stacked.png")
plt.close()
print("Saved D1_entity_type_stacked.png")


# ═══════════════════════════════════════════════════════════════════════════
# D2. Top companies & universities (dot charts)
# ═══════════════════════════════════════════════════════════════════════════
# Use unique family-level han_name: split the first entry of han_name on comma
# For 'Company-only' patents (psn_sector == 'COMPANY' exactly), the first
# han_name entry is typically a clean company name.
companies_raw = (
    green
    .filter(pl.col("psn_sector") == "COMPANY")
    .with_columns(
        pl.col("han_name").str.split(",").list.first().str.strip_chars().alias("entity")
    )
    .unique(subset=["docdb_family_id", "entity"])
    .group_by("entity")
    .agg([
        pl.len().alias("n_families"),
        pl.col("nb_citing_docdb_fam").sum().alias("total_cit"),
        pl.col("appln_auth").mode().first().alias("country"),
        pl.col("primary_sector").mode().first().alias("main_sector"),
    ])
    .filter(pl.col("entity").str.len_chars() > 4)
    .sort("n_families", descending=True)
    .head(20)
    .to_pandas()
)

univs_raw = (
    green
    .filter(pl.col("psn_sector").str.contains("UNIVERSITY"))
    .with_columns(
        pl.col("han_name").str.split(",").list.last().str.strip_chars().alias("entity")
    )
    .unique(subset=["docdb_family_id", "entity"])
    .group_by("entity")
    .agg([
        pl.len().alias("n_families"),
        pl.col("nb_citing_docdb_fam").sum().alias("total_cit"),
        pl.col("appln_auth").mode().first().alias("country"),
        pl.col("primary_sector").mode().first().alias("main_sector"),
    ])
    .filter(pl.col("entity").str.len_chars() > 4)
    .filter(
        pl.col("entity").str.to_uppercase().str.contains(
            "UNIV|INSTIT|COLLEGE|ACADEM|SCHOOL|TECH|POLITEC"
        )
    )
    .sort("n_families", descending=True)
    .head(20)
    .to_pandas()
)

fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(16, 6))

def dot_chart(ax, df, label, color_field="main_sector"):
    df = df.head(15).copy()
    df["entity_short"] = df["entity"].str[:35]
    y = np.arange(len(df))
    # Dot size ~ total citations
    sizes = (df["total_cit"] / df["total_cit"].max() * 400 + 30).values
    colors = [SECTOR_COLOR.get(s, "#aaa") for s in df.get("main_sector", [""] * len(df))]
    ax.scatter(df["n_families"], y, s=sizes, c=colors, zorder=3, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["entity_short"], fontsize=8)
    ax.set_xlabel("Unique green patent families", fontsize=9)
    ax.set_title(label, fontsize=10, pad=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    # Legend for sectors
    handles = [mpatches.Patch(color=SECTOR_COLOR.get(s, "#aaa"), label=s) for s in SECTORS_ORDERED if s in df.get("main_sector", []).values]
    ax.legend(handles=handles[:5], fontsize=7, loc="lower right")
    ax.invert_yaxis()

dot_chart(ax_c, companies_raw, "Top 15 Companies — Green Patent Families")
dot_chart(ax_u, univs_raw,   "Top 15 Universities — Green Patent Families")

# Global legend: dot size = citations
for size, label in [(50, "Low cit."), (200, "Med cit."), (430, "High cit.")]:
    ax_c.scatter([], [], s=size, color="#aaa", alpha=0.7, label=label)
ax_c.legend(fontsize=7, title="Citation weight", title_fontsize=7)

fig.suptitle(
    "Top Green Patent Applicants — Harmonised Entities (TW excluded)",
    fontsize=12, y=1.01,
)
fig.tight_layout()
fig.savefig(f"{OUT}/D2_top_applicants.png", bbox_inches="tight")
plt.close()
print("Saved D2_top_applicants.png")


# ═══════════════════════════════════════════════════════════════════════════
# D3. Sector × entity-type heatmap (% share)
# ═══════════════════════════════════════════════════════════════════════════
sec_ent = (
    green
    .filter(pl.col("primary_sector").is_in(SECTORS_ORDERED))
    .group_by(["primary_sector", "entity_type"])
    .agg(pl.len().alias("n"))
    .to_pandas()
    .pivot(index="primary_sector", columns="entity_type", values="n")
    .fillna(0)
    .reindex(index=SECTORS_ORDERED,
             columns=[c for c in ENTITY_ORDER if c in ["Company","University","Gov/Non-profit","Individual"]])
    .fillna(0)
)
sec_ent_pct = sec_ent.div(sec_ent.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(sec_ent_pct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(sec_ent_pct.columns)))
ax.set_yticks(range(len(sec_ent_pct.index)))
ax.set_xticklabels(sec_ent_pct.columns, fontsize=10)
ax.set_yticklabels(sec_ent_pct.index, fontsize=10)
for i in range(len(sec_ent_pct.index)):
    for j in range(len(sec_ent_pct.columns)):
        v = sec_ent_pct.values[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                color="white" if v > 55 else "black")
plt.colorbar(im, ax=ax, shrink=0.8, label="Share within sector (%)")
plt.title("Applicant Type Mix by Green Technology Sector", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/D3_sector_entity_heatmap.png")
plt.close()
print("Saved D3_sector_entity_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# D4. Country × entity-type for top countries
# ═══════════════════════════════════════════════════════════════════════════
TOP_CTRY = ["US", "CN", "JP", "DE", "KR", "GB", "FR", "DK", "AU", "CA"]
ctry_ent = (
    green
    .filter(pl.col("appln_auth").is_in(TOP_CTRY))
    .group_by(["appln_auth", "entity_type"])
    .agg(pl.len().alias("n"))
    .to_pandas()
    .pivot(index="appln_auth", columns="entity_type", values="n")
    .fillna(0)
    .reindex(index=TOP_CTRY,
             columns=[c for c in ENTITY_ORDER if c in ["Company","University","Gov/Non-profit","Individual"]])
    .fillna(0)
)
ctry_pct = ctry_ent.div(ctry_ent.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(len(TOP_CTRY))
for etype in ["Company", "University", "Gov/Non-profit", "Individual"]:
    if etype not in ctry_pct.columns:
        continue
    vals = ctry_pct[etype].values
    ax.bar(TOP_CTRY, vals, bottom=bottom, label=etype,
           color=ENTITY_COLOR[etype], alpha=0.90, width=0.7)
    # Label segments > 8%
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 8:
            ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="white" if v > 20 else "black")
    bottom += vals

ax.set_ylabel("Share of green applications (%)", fontsize=10)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(fontsize=9, loc="upper right")
plt.title("Applicant Type Composition per Country (TW excluded)", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/D4_country_entity_type.png")
plt.close()
print("Saved D4_country_entity_type.png")


# ═══════════════════════════════════════════════════════════════════════════
# E1. Influential patent bubble chart
#     X = forward citations (5yr)  |  Y = OECD quality index
#     Size = family size           |  Color = primary sector
#     (Dechezlepretre-style scatter)
# ═══════════════════════════════════════════════════════════════════════════
green_fam = green.unique(subset=["docdb_family_id"])
green_q = (
    green_fam
    .join(pq, on="docdb_family_id", how="inner")
    .join(pr, on="docdb_family_id", how="left")
    .filter(pl.col("fwd_citations_5yr") > 0)
    .filter(pl.col("patent_quality_index_4").is_not_null())
    .filter(pl.col("primary_sector").is_in(SECTORS_ORDERED))
    .sort("fwd_citations_5yr", descending=True)
    .head(200000)   # use top-200k to keep plot readable
    .to_pandas()
)

# Winsorise at 99th pct for display
c99 = np.percentile(green_q["fwd_citations_5yr"], 99)
q99 = np.percentile(green_q["patent_quality_index_4"], 99)
green_q_plot = green_q[(green_q["fwd_citations_5yr"] <= c99) &
                        (green_q["patent_quality_index_4"] <= q99)].copy()

# Top-30 high-influence labelled points
top30 = (
    green.unique(subset=["docdb_family_id"])
    .join(pq, on="docdb_family_id", how="inner")
    .join(pr, on="docdb_family_id", how="left")
    .filter(pl.col("category") == "H")
    .sort("pagerank", descending=True)
    .head(30)
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(12, 8))

for sec in SECTORS_ORDERED:
    mask = green_q_plot["primary_sector"] == sec
    sub = green_q_plot[mask]
    sz = np.clip(sub["family_size"].values * 1.5, 3, 60)
    ax.scatter(sub["fwd_citations_5yr"], sub["patent_quality_index_4"],
               s=sz, c=SECTOR_COLOR[sec], alpha=0.18, rasterized=True, label=sec)

# Overlay top-30 H patents
for _, row in top30.iterrows():
    sec = str(row.get("primary_sector", "Energy")).split(",")[0].strip()
    c = SECTOR_COLOR.get(sec, "#264653")
    ax.scatter(row["fwd_citations_5yr"], row["patent_quality_index_4"],
               s=max(30, row.get("family_size", 5) * 3), c=c, alpha=0.95,
               edgecolors="black", linewidths=0.8, zorder=5)

# Label the very top-10 by PageRank
top10_label = top30.sort_values("pagerank", ascending=False).head(10)
for _, row in top10_label.iterrows():
    title = str(row.get("appln_title", ""))[:28] + "…" if len(str(row.get("appln_title", ""))) > 28 else str(row.get("appln_title", ""))
    ax.annotate(
        f"{row['appln_auth']} {row['yr']}\n{title}",
        xy=(row["fwd_citations_5yr"], row["patent_quality_index_4"]),
        xytext=(10, 5), textcoords="offset points",
        fontsize=6.5, color="#222",
        arrowprops=dict(arrowstyle="-", color="#999", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85),
        zorder=6,
    )

ax.set_xlabel("5-year forward citations (quality signal)", fontsize=10)
ax.set_ylabel("OECD Patent Quality Index (0–1)", fontsize=10)
ax.set_xlim(0)
ax.set_ylim(0, 1.05)

# Quadrant lines
ax.axvline(np.percentile(green_q["fwd_citations_5yr"], 75), color="#ccc", lw=0.8, ls="--")
ax.axhline(np.percentile(green_q["patent_quality_index_4"], 75), color="#ccc", lw=0.8, ls="--")
ax.text(c99 * 0.72, 0.97, "High-influence zone", fontsize=8, color="#888", ha="right")

handles = [mpatches.Patch(color=SECTOR_COLOR[s], label=s) for s in SECTORS_ORDERED]
ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left",
          title="Sector (dot size = family size)", title_fontsize=7.5)

plt.title(
    "Most Influential Green Patents — Citations × OECD Quality × Family Size\n"
    "Black-bordered dots = top-30 High-influence (H) patents by PageRank",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/E1_influential_scatter.png", dpi=150)
plt.close()
print("Saved E1_influential_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# E2. Top-20 influential patents table (combined score)
# ═══════════════════════════════════════════════════════════════════════════
top20_combined = (
    green.unique(subset=["docdb_family_id"])
    .join(pq, on="docdb_family_id", how="inner")
    .join(pr, on="docdb_family_id", how="left")
    .filter(pl.col("category") == "H")
    .filter(pl.col("yr") <= 2021)
    .with_columns([
        # Normalise each metric to 0-1, then combine
        (pl.col("fwd_citations_5yr") / pl.col("fwd_citations_5yr").max()).alias("n_cit"),
        pl.col("patent_quality_index_4").alias("n_q"),
        (pl.col("pagerank") / pl.col("pagerank").max()).alias("n_pr"),
        (pl.col("family_size") / pl.col("family_size").max()).alias("n_fam"),
    ])
    .with_columns(
        ((pl.col("n_cit") * 0.35 + pl.col("n_q") * 0.30
          + pl.col("n_pr") * 0.25 + pl.col("n_fam") * 0.10)).alias("composite")
    )
    .sort("composite", descending=True)
    .head(20)
    .select(["docdb_family_id", "appln_auth", "yr", "primary_sector",
             "fwd_citations_5yr", "patent_quality_index_4", "pagerank",
             "family_size", "composite", "appln_title"])
    .to_pandas()
)
top20_combined.to_csv(f"{OUT}/E2_top20_influential.csv", index=False)
print("Saved E2_top20_influential.csv")

# Plot top-20 as horizontal dot chart with multi-metric lollipop
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
metrics = [
    ("fwd_citations_5yr",       "5yr Forward Citations",    "#2a9d8f"),
    ("patent_quality_index_4",  "OECD Quality Index (0–1)", "#264653"),
    ("pagerank",                "PageRank Centrality",       "#e76f51"),
]
labels = (top20_combined["appln_auth"].str[:2] + " "
          + top20_combined["yr"].astype(str) + " – "
          + top20_combined["appln_title"].str[:30]).values

for ax, (col, title, color) in zip(axes, metrics):
    vals = top20_combined[col].values
    y = np.arange(len(vals))
    ax.barh(y, vals, color=color, height=0.6, alpha=0.82)
    ax.set_title(title, fontsize=9)
    ax.invert_yaxis()
    if ax == axes[0]:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", labelsize=7.5)

# Color rows by sector
sector_colors = [SECTOR_COLOR.get(s.split(",")[0].strip(), "#aaa") for s in top20_combined["primary_sector"]]
for i, sc in enumerate(sector_colors):
    for ax in axes:
        ax.axhspan(i - 0.45, i + 0.45, alpha=0.06, color=sc, zorder=0)

fig.suptitle("Top-20 Most Influential Green Patents (Composite Score)\nCitations × OECD Quality × PageRank × Family Size",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/E2_top20_influential.png", bbox_inches="tight")
plt.close()
print("Saved E2_top20_influential.png")


# ═══════════════════════════════════════════════════════════════════════════
# F1. PageRank distribution by sector (violin + strip)
#     Dechezlepretre 2015 style: compare influence concentration
# ═══════════════════════════════════════════════════════════════════════════
green_pr_joined = (
    green.unique(subset=["docdb_family_id"])
    .join(pr, on="docdb_family_id", how="inner")
    .filter(pl.col("primary_sector").is_in(SECTORS_ORDERED))
    .filter(pl.col("pagerank") > 0)
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(12, 6))
positions = np.arange(len(SECTORS_ORDERED))

for i, sec in enumerate(SECTORS_ORDERED):
    sub = green_pr_joined[green_pr_joined["primary_sector"] == sec]["pagerank"].values
    log_sub = np.log10(sub + 1e-12)
    # Violin
    vp = ax.violinplot([log_sub], positions=[i], widths=0.75,
                       showmedians=True, showextrema=False)
    for pc in vp["bodies"]:
        pc.set_facecolor(SECTOR_COLOR[sec])
        pc.set_alpha(0.55)
        pc.set_edgecolor(SECTOR_COLOR[sec])
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.5)
    # Strip of top-1% dots
    top1 = np.percentile(log_sub, 99)
    elite = log_sub[log_sub >= top1]
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(elite))
    ax.scatter(np.full(len(elite), i) + jitter, elite,
               s=8, color=SECTOR_COLOR[sec], alpha=0.6, zorder=3)

ax.set_xticks(positions)
ax.set_xticklabels(SECTORS_ORDERED, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("log₁₀(PageRank)", fontsize=10)
ax.axhline(np.log10(1e-6), color="#bbb", lw=0.7, ls="--")
ax.text(len(SECTORS_ORDERED) - 0.5, np.log10(1e-6) + 0.05,
        "avg node ≈ 10⁻⁶", fontsize=7.5, color="#888", ha="right")
plt.title(
    "PageRank Distribution by Green Technology Sector\n"
    "Dots = top-1% most influential patents (Dechezleprêtre 2015 approach)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/F1_pagerank_by_sector.png")
plt.close()
print("Saved F1_pagerank_by_sector.png")


# ═══════════════════════════════════════════════════════════════════════════
# F2. Innovation-influence Lorenz curve by sector
#     Cumulative share of PageRank mass vs. cumulative share of patents
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

gini_data = []
for sec in SECTORS_ORDERED:
    sub = green_pr_joined[green_pr_joined["primary_sector"] == sec]["pagerank"].values
    sub_sorted = np.sort(sub)
    cum_patents = np.linspace(0, 1, len(sub_sorted) + 1)
    cum_pr = np.concatenate([[0], np.cumsum(sub_sorted) / sub_sorted.sum()])
    gini = 1 - 2 * np.trapz(cum_pr, cum_patents)
    gini_data.append((sec, gini))
    ax.plot(cum_patents, cum_pr, color=SECTOR_COLOR[sec], lw=1.8,
            label=f"{sec} (Gini={gini:.2f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect equality")
ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="black")
ax.set_xlabel("Cumulative share of patents (poorest first)", fontsize=10)
ax.set_ylabel("Cumulative share of PageRank mass", fontsize=10)
ax.legend(fontsize=8.5, loc="upper left")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.title(
    "Innovation-Influence Lorenz Curves — Green Technology Sectors\n"
    "Steeper = more concentrated influence in fewer patents (higher Gini)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/F2_lorenz_curve.png")
plt.close()
print("Saved F2_lorenz_curve.png")
print("  Gini coefficients:", [(s, f"{g:.3f}") for s, g in gini_data])


# ═══════════════════════════════════════════════════════════════════════════
# F3. Sankey-style citation flow diagram  H / G / N
# ═══════════════════════════════════════════════════════════════════════════
# Count forward-citation flows
cat_flow = (
    cit.filter(pl.col("direction") == "forward")
    .group_by(["citing_category", "cited_category"])
    .agg(pl.len().alias("n"))
    .to_pandas()
)
cats = ["H", "G", "N"]
cat_label = {"H": "High-influence\ngreen (H)", "G": "Green (G)", "N": "Neighbor (N)"}
cat_color = {"H": "#e76f51", "G": "#2a9d8f", "N": "#264653"}

# Layout: left = citing, right = cited
# Heights proportional to total in/out flow
out_flows = cat_flow.groupby("citing_category")["n"].sum()
in_flows  = cat_flow.groupby("cited_category")["n"].sum()

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.05, 1.05)
ax.axis("off")

TOTAL = cat_flow["n"].sum()
GAP = 0.05

def build_offsets(flows_dict, keys):
    total = sum(flows_dict.get(k, 0) for k in keys)
    heights = {k: flows_dict.get(k, 0) / total for k in keys}
    offsets = {}
    y = 0
    for k in keys:
        offsets[k] = y
        y += heights[k] + GAP
    # renormalise to [0,1]
    scale = 1.0 / (y - GAP)
    return {k: offsets[k] * scale for k in keys}, {k: heights[k] * scale for k in keys}

left_offsets,  left_heights  = build_offsets(out_flows.to_dict(), cats)
right_offsets, right_heights = build_offsets(in_flows.to_dict(),  cats)

# Draw node bars
bar_w = 0.04
for c in cats:
    ax.add_patch(mpatches.FancyBboxPatch(
        (-bar_w, left_offsets[c]),  bar_w, left_heights[c],
        boxstyle="square,pad=0", fc=cat_color[c], ec="none", alpha=0.9))
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.0,    right_offsets[c]), bar_w, right_heights[c],
        boxstyle="square,pad=0", fc=cat_color[c], ec="none", alpha=0.9))
    # Labels
    ax.text(-bar_w - 0.02, left_offsets[c] + left_heights[c]/2,
            cat_label[c], ha="right", va="center", fontsize=9, fontweight="bold",
            color=cat_color[c])
    ax.text(1.0 + bar_w + 0.02, right_offsets[c] + right_heights[c]/2,
            cat_label[c], ha="left", va="center", fontsize=9, fontweight="bold",
            color=cat_color[c])

# Draw flow ribbons using bezier curves
# Track current y-position within each node bar
left_cursor  = {c: left_offsets[c]  for c in cats}
right_cursor = {c: right_offsets[c] for c in cats}

for _, row in cat_flow.sort_values("n", ascending=False).iterrows():
    src, dst, n = row["citing_category"], row["cited_category"], row["n"]
    w = (n / TOTAL) / (1 + 3 * GAP)   # ribbon height (proportional)
    src_y = left_cursor[src]
    dst_y = right_cursor[dst]
    left_cursor[src]  += w
    right_cursor[dst] += w
    # Bezier ribbon
    verts = [
        (0.0, src_y), (0.0, src_y + w),
        (0.5, dst_y + w), (1.0, dst_y + w),
        (1.0, dst_y), (0.5, dst_y),
        (0.0, src_y),
    ]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    # Smooth with cubic interpolation
    from matplotlib.path import Path
    codes = ([Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
              Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
             if False else  # fallback to filled polygon
             [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY])
    # Simple filled poly
    alpha_val = min(0.65, n / TOTAL * 30 + 0.10)
    # Use source color blended with dest color
    base_c = cat_color[src]
    ax.fill(xs, ys, color=base_c, alpha=alpha_val)
    # Annotation
    mid_x = 0.5
    mid_y = (src_y + dst_y) / 2 + w / 2
    if n / TOTAL > 0.05:
        ax.text(mid_x, mid_y, f"{n/1e6:.1f}M", ha="center", va="center",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=base_c, ec="none", alpha=0.8))

ax.text(0.0, 1.02, "Citing (knowledge\nrecipient)", ha="center", fontsize=9, color="#555")
ax.text(1.0, 1.02, "Cited (knowledge\nsource)",    ha="center", fontsize=9, color="#555")
ax.set_title(
    "Citation Flow Sankey: High-influence (H) · Green (G) · Neighbor (N)\n"
    "Width proportional to citation count · Total = 22.5M citation edges",
    fontsize=11, pad=14,
)
fig.tight_layout()
fig.savefig(f"{OUT}/F3_sankey_hgn.png", bbox_inches="tight")
plt.close()
print("Saved F3_sankey_hgn.png")


# ═══════════════════════════════════════════════════════════════════════════
# F4. Sector-to-sector cross-citation heatmap
#     (Dechezlepretre: "clean→clean", "dirty→clean" analogously we do
#     sector→sector to see which green sectors build on which)
# ═══════════════════════════════════════════════════════════════════════════
# Join citing and cited family to sector
fam_sector = (
    green.unique(subset=["docdb_family_id"])
    .select(["docdb_family_id", "primary_sector"])
    .filter(pl.col("primary_sector").is_in(SECTORS_ORDERED))
)

cit_fwd = cit.filter(pl.col("direction") == "forward")
sec_flow = (
    cit_fwd
    .join(fam_sector.rename({"primary_sector": "citing_sec"}), on="docdb_family_id", how="inner")
    .join(
        fam_sector.rename({"primary_sector": "cited_sec",
                           "docdb_family_id": "cited_docdb_family_id"}),
        on="cited_docdb_family_id", how="inner",
    )
    .group_by(["citing_sec", "cited_sec"])
    .agg(pl.len().alias("n"))
    .to_pandas()
    .pivot(index="citing_sec", columns="cited_sec", values="n")
    .reindex(index=SECTORS_ORDERED, columns=SECTORS_ORDERED)
    .fillna(0).astype(int)
)

# Row-normalise to get citation-sending shares
sec_flow_norm = sec_flow.div(sec_flow.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(sec_flow_norm.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=50)
ax.set_xticks(range(len(SECTORS_ORDERED)))
ax.set_yticks(range(len(SECTORS_ORDERED)))
ax.set_xticklabels([s[:8] for s in SECTORS_ORDERED], fontsize=8.5, rotation=30, ha="right")
ax.set_yticklabels(SECTORS_ORDERED, fontsize=8.5)
ax.set_xlabel("Cited sector (knowledge source)", fontsize=10)
ax.set_ylabel("Citing sector (knowledge recipient)", fontsize=10)
for i in range(len(SECTORS_ORDERED)):
    for j in range(len(SECTORS_ORDERED)):
        v = sec_flow_norm.values[i, j]
        if v > 3:
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 30 else "black")
cb = plt.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Share of citing sector's citations (%)", fontsize=8)
plt.title(
    "Green Technology Sector Knowledge Flows (Citation Matrix)\n"
    "Row = citing sector · Column = cited sector · Row-normalised (%)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/F4_sector_citation_matrix.png")
plt.close()
print("Saved F4_sector_citation_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════
# F5. PageRank × HITS authority: sector comparison scatter
# ═══════════════════════════════════════════════════════════════════════════
green_pr_hits = (
    green.unique(subset=["docdb_family_id"])
    .join(pr, on="docdb_family_id", how="inner")
    .join(hits.select(["docdb_family_id","authority_score"]), on="docdb_family_id", how="left")
    .filter(pl.col("primary_sector").is_in(SECTORS_ORDERED))
    .filter(pl.col("pagerank") > 1e-9)
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 7))
for sec in SECTORS_ORDERED:
    sub = green_pr_hits[green_pr_hits["primary_sector"] == sec]
    sub_s = sub.sample(min(5000, len(sub)), random_state=42)
    ax.scatter(
        np.log10(sub_s["pagerank"] + 1e-12),
        np.log10(sub_s["authority_score"] + 1e-65),
        s=5, c=SECTOR_COLOR[sec], alpha=0.25, rasterized=True, label=sec,
    )

# Add sector centroids
for sec in SECTORS_ORDERED:
    sub = green_pr_hits[green_pr_hits["primary_sector"] == sec]
    cx = np.log10(sub["pagerank"].mean() + 1e-12)
    cy = np.log10(sub["authority_score"].mean() + 1e-65)
    ax.scatter([cx], [cy], s=200, c=SECTOR_COLOR[sec], marker="D",
               edgecolors="black", linewidths=1, zorder=5)
    ax.text(cx, cy + 0.5, sec[:6], ha="center", fontsize=8, fontweight="bold",
            color=SECTOR_COLOR[sec])

handles = [mpatches.Patch(color=SECTOR_COLOR[s], label=s) for s in SECTORS_ORDERED]
ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower right")
ax.set_xlabel("log₁₀(PageRank)  — overall network influence", fontsize=10)
ax.set_ylabel("log₁₀(HITS Authority Score)  — knowledge source quality", fontsize=10)
plt.title(
    "PageRank vs. HITS Authority by Green Technology Sector\n"
    "Diamonds = sector centroids · Dechezleprêtre 2015 network science approach",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/F5_pagerank_hits_scatter.png", dpi=150)
plt.close()
print("Saved F5_pagerank_hits_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("APPLICANT & INFLUENCE SUMMARY (TW excluded)")
print("=" * 65)
ent_counts = green["entity_type"].value_counts().to_pandas().set_index("entity_type")
for e in ENTITY_ORDER:
    n = ent_counts.get("count", ent_counts.get("len", pd.Series())).get(e, 0) if e in ent_counts.index else 0
print("Applicant breakdown (% of all applications):")
tot = green.height
for e in ENTITY_ORDER:
    n = green.filter(pl.col("entity_type") == e).height
    print(f"  {e:20s}: {n:>9,}  ({n/tot*100:.1f}%)")
print(f"\nTop gini (most concentrated): {max(gini_data, key=lambda x: x[1])}")
print(f"Bot gini (most distributed):  {min(gini_data, key=lambda x: x[1])}")
print(f"\nAll figures saved to {OUT}/")
