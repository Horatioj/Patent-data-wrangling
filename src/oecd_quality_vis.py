"""
OECD Patent Quality Index (PQI) visualisation — green patents, 1985–2025.

The OECD modified-HHI PQI combines forward citations, family size, claims and
generality into a 0–1 composite (OECD 2009 "Patent Quality Indicators").  Values
are inventor-country fractional (consistent with the fractional maps and the
green_patent_vis.py / stacked-histogram charts): a family is credited to its
inventor countries by inventor share, and country PQI is the inventor-share
weighted mean over its families.

Inputs (produced by src/count_pat_q_per_year.py and src/patent_quality.py):
  • oecd_patent_quality_country_year_complete.csv   (country × year panel)
  • oecd_patent_quality_country_rankings.csv         (country overall ranking)

Figures:
  Q1 — PQI trajectory over time for key inventor countries (line)
  Q2 — Cross-country OECD PQI ranking, top 20 by volume (horizontal bar)
  Q3 — Quality vs. quantity scatter (PQI vs. fractional green volume)

Run from the project root:
    python src/oecd_quality_vis.py
"""

import os
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

YEAR_START, YEAR_END = 1985, 2025
# Citations need time to accrue: trajectories are shown only through CITE_MATURE.
CITE_MATURE = 2021
# Minimum fractional green families for a country-year / country to be plotted.
MIN_VOL_YR = 30
MIN_VOL_RANK = 500

PQI_COL = "inventor_fractional_pqi_oecd"   # inventor-share weighted OECD PQI
VOL_COL = "inventor_frac_patents"          # inventor-fractional green families

KEY_CTRY = ["US", "CN", "JP", "DE", "KR", "GB", "FR", "DK"]
CTRY_LABEL = {
    "US": "United States", "CN": "China", "JP": "Japan", "DE": "Germany",
    "KR": "South Korea", "GB": "United Kingdom", "FR": "France", "DK": "Denmark",
}
CTRY_COLOR = {
    "US": "#4a90d9", "CN": "#e05252", "JP": "#f4a261", "DE": "#6a4c93",
    "KR": "#2a9d8f", "GB": "#264653", "FR": "#e9c46a", "DK": "#81b29a",
}
FRAC_NOTE = "Inventor-country fractional · OECD modified-HHI PQI"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading OECD quality country-year panel …")
cy = (
    pl.read_csv(f"{DATA}/oecd_patent_quality_country_year_complete.csv")
    .select(["countries", "family_year", PQI_COL, VOL_COL])
    .filter((pl.col("family_year") >= YEAR_START) & (pl.col("family_year") <= YEAR_END))
    .to_pandas()
)

print("Loading OECD quality country rankings …")
rank = (
    pl.read_csv(f"{DATA}/oecd_patent_quality_country_rankings.csv")
    .select(["countries", PQI_COL, VOL_COL])
    .filter(pl.col(VOL_COL).is_not_null() & (pl.col(PQI_COL) > 0))
    .to_pandas()
)
print("Loaded.\n")


# ════════════════════════════════════════════════════════════════════════════
# Q1 – PQI trajectory over time for key inventor countries
# ════════════════════════════════════════════════════════════════════════════
years = np.arange(YEAR_START, CITE_MATURE + 1)
fig, ax = plt.subplots(figsize=(13, 6))
for c in KEY_CTRY:
    sub = cy[(cy["countries"] == c) & (cy["family_year"] <= CITE_MATURE)
             & (cy[VOL_COL] >= MIN_VOL_YR)].set_index("family_year")
    if len(sub) < 5:
        continue
    s = sub[PQI_COL].reindex(years)
    ax.plot(years, s.values, color=CTRY_COLOR[c], lw=2.0, marker="o",
            markersize=3, label=CTRY_LABEL[c])
ax.set_xlim(YEAR_START, CITE_MATURE)
ax.set_xlabel("Earliest family filing year", fontsize=10)
ax.set_ylabel("OECD Patent Quality Index (0–1)", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9,
          frameon=False)
ax.text(YEAR_START + 0.4, ax.get_ylim()[1] * 0.02,
        f"Years with ≥{MIN_VOL_YR} fractional green families; shown through "
        f"{CITE_MATURE} (citation maturity)", fontsize=7.5, color="#888")
plt.title(f"OECD Patent Quality Index Trajectory by Inventor Country, "
          f"1985–{CITE_MATURE}\n({FRAC_NOTE})", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/Q1_oecd_pqi_trajectory.png", bbox_inches="tight")
plt.close()
print("Saved Q1_oecd_pqi_trajectory.png")


# ════════════════════════════════════════════════════════════════════════════
# Q2 – Cross-country OECD PQI ranking (top 20 by volume)
# ════════════════════════════════════════════════════════════════════════════
top = (rank[rank[VOL_COL] >= MIN_VOL_RANK]
       .sort_values(PQI_COL, ascending=False).head(20).iloc[::-1])
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(top))
bar_colors = [CTRY_COLOR.get(c, "#9aa7b1") for c in top["countries"]]
bars = ax.barh(y, top[PQI_COL].values, color=bar_colors, height=0.7)
ax.set_yticks(y)
ax.set_yticklabels([CTRY_LABEL.get(c, c) for c in top["countries"]], fontsize=9)
ax.set_xlabel("OECD Patent Quality Index (0–1)", fontsize=10)
ax.set_xlim(0, max(top[PQI_COL].max() * 1.18, 0.1))
for bar, pqi, vol in zip(bars, top[PQI_COL].values, top[VOL_COL].values):
    ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{pqi:.3f}  ({vol/1000:.0f}k fam)", va="center", fontsize=8, color="#444")
ax.set_axisbelow(True)
ax.xaxis.grid(True); ax.yaxis.grid(False)
plt.title(f"Green Patent Quality by Inventor Country — Top 20 by Volume\n"
          f"Countries with ≥{MIN_VOL_RANK:,} fractional green families · {FRAC_NOTE}",
          fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/Q2_oecd_pqi_ranking.png", bbox_inches="tight")
plt.close()
print("Saved Q2_oecd_pqi_ranking.png")


# ════════════════════════════════════════════════════════════════════════════
# Q3 – Quality vs. quantity scatter (PQI vs. fractional green volume)
# ════════════════════════════════════════════════════════════════════════════
sc = rank[rank[VOL_COL] >= 100].copy()
fig, ax = plt.subplots(figsize=(11, 7))
sizes = np.clip(sc[VOL_COL].values / sc[VOL_COL].max() * 1400 + 12, 12, 1400)
is_key = sc["countries"].isin(KEY_CTRY)
ax.scatter(sc.loc[~is_key, VOL_COL], sc.loc[~is_key, PQI_COL],
           s=sizes[~is_key.values], color="#b8c2cc", alpha=0.55,
           edgecolors="white", linewidths=0.4, zorder=2)
for c in KEY_CTRY:
    r = sc[sc["countries"] == c]
    if r.empty:
        continue
    ax.scatter(r[VOL_COL], r[PQI_COL],
               s=np.clip(r[VOL_COL].values / sc[VOL_COL].max() * 1400 + 60, 60, 1400),
               color=CTRY_COLOR[c], alpha=0.9, edgecolors="black", linewidths=0.8,
               zorder=4, label=CTRY_LABEL[c])
    ax.annotate(c, xy=(r[VOL_COL].values[0], r[PQI_COL].values[0]),
                xytext=(6, 6), textcoords="offset points", fontsize=9,
                fontweight="bold", color="#222", zorder=5)

# Median reference lines → quality/quantity quadrants
med_vol = sc[VOL_COL].median()
med_pqi = sc[PQI_COL].median()
ax.axvline(med_vol, color="#ccc", lw=0.9, ls="--")
ax.axhline(med_pqi, color="#ccc", lw=0.9, ls="--")
ax.set_xscale("log")
ax.set_xlabel("Fractional green patent families (log scale)", fontsize=10)
ax.set_ylabel("OECD Patent Quality Index (0–1)", fontsize=10)
ax.text(0.99, 0.98, "High volume · High quality", transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="#999")
ax.text(0.01, 0.98, "Low volume · High quality", transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color="#999")
ax.legend(fontsize=8, loc="lower right", ncol=2, title="Bubble size = volume",
          title_fontsize=8)
plt.title(f"Green Patent Quality vs. Quantity by Inventor Country\n"
          f"All countries with ≥100 fractional green families · {FRAC_NOTE}",
          fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/Q3_oecd_quality_vs_quantity.png")
plt.close()
print("Saved Q3_oecd_quality_vs_quantity.png")

print(f"\nAll OECD PQI figures saved to: {OUT}/")
