"""
Green patent statistics and visualisation — extended analysis (1985–2025).

Covers:
  A. Time-series analysis (Hall 2005 / Griliches 1990 style)
       • Log-linear growth trend with OLS fit
       • Year-on-year growth rates
       • Citation-weighted quality index
       • Technology diffusion (cumulative stock)
  B. Multi-country comparison
       • Green patent intensity (green / total granted) per country × year
       • Country share of global green patents
       • Country ranking dynamics
  C. Knowledge-spillover visualisation via patent citations
       • Cross-country citation flow heatmap
       • Domestic vs. international citation ratios
       • US–China bilateral citation flows

Run from the project root:
    python src/green_patent_vis.py

References (methodological):
  Hall, B.H. (2005). "Measuring the Returns to R&D." NBER Working Paper 11224.
  Griliches, Z. (1990). "Patent Statistics as Economic Indicators: A Survey."
      Journal of Economic Literature, 28(4), 1661–1707.
  Hall, B.H., Jaffe, A., Trajtenberg, M. (2001). "The NBER Patent Citation Data
      File." NBER Working Paper 8498.
  Dechezleprêtre, A. et al. (2011). "Invention and Transfer of Climate Change
      Mitigation Technologies." Review of Environmental Economics and Policy.
"""

import os
import sys
import textwrap

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "axes.prop_cycle": matplotlib.cycler(color=[
        "#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51",
        "#6a4c93","#4a90d9","#81b29a","#a8dadc","#e05252",
        "#c77dff","#f77f00","#d62828","#457b9d",
    ]),
})

OUT = "PATSTAT2025FALL/output/vis"
os.makedirs(OUT, exist_ok=True)

YEAR_START, YEAR_END = 1985, 2025
# Flag years where filing lag makes counts incomplete
INCOMPLETE_FROM = 2022

KEY_CTRY = ["US", "CN", "JP", "DE", "KR", "EP", "GB", "FR", "DK", "TW", "AU", "CA", "NL"]
CTRY_LABEL = {
    "US": "United States", "CN": "China", "JP": "Japan", "DE": "Germany",
    "KR": "South Korea", "EP": "European Patent Office", "GB": "United Kingdom",
    "FR": "France", "DK": "Denmark", "TW": "Taiwan", "AU": "Australia",
    "CA": "Canada", "NL": "Netherlands",
}
CTRY_COLOR = {
    "US": "#4a90d9", "CN": "#e05252", "JP": "#f4a261", "DE": "#6a4c93",
    "KR": "#2a9d8f", "EP": "#a8a8a8", "GB": "#264653", "FR": "#e9c46a",
    "DK": "#81b29a", "TW": "#c77dff", "AU": "#f77f00", "CA": "#a8dadc",
    "NL": "#d62828",
}


# ════════════════════════════════════════════════════════════════════════════
# 0. Load data
# ════════════════════════════════════════════════════════════════════════════
print("Loading green_patent8526.parquet …")
green = pl.read_parquet(
    "PATSTAT2025FALL/output/green_patent8526.parquet",
    columns=[
        "appln_filing_date", "appln_auth", "sector", "mitigation_adaptation",
        "docdb_family_id", "nb_citing_docdb_fam", "docdb_family_size", "psn_sector",
    ],
).with_columns(
    pl.col("appln_filing_date").str.slice(0, 4).cast(pl.Int32).alias("yr"),
)

print("Loading docdb_family_year.parquet …")
total_yr_df = (
    pl.scan_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet")
    .filter((pl.col("family_year") >= YEAR_START) & (pl.col("family_year") <= YEAR_END))
    .group_by("family_year").agg(pl.len().alias("total"))
    .sort("family_year").collect()
    .to_pandas().set_index("family_year")["total"]
)

print("Loading tls201.parquet (country totals) …")
total_ctry_yr_wide = (
    pl.scan_parquet("PATSTAT2025FALL/output/tls201.parquet")
    .filter(
        (pl.col("year") >= YEAR_START) & (pl.col("year") <= YEAR_END)
        & (pl.col("granted") == "Y")
        & pl.col("appln_auth").is_in(KEY_CTRY)
    )
    .group_by(["year", "appln_auth"]).agg(pl.len().alias("total"))
    .sort(["year", "appln_auth"]).collect()
    .to_pandas()
    .pivot(index="year", columns="appln_auth", values="total")
    .fillna(0).astype(int)
)

print("Loading citation_edges_categorized.parquet …")
cit_edges = pl.read_parquet("PATSTAT2025FALL/output/citation_edges_categorized.parquet")

print("Loading docdb_family_country.parquet …")
fam_ctry = pl.read_parquet("PATSTAT2025FALL/output/docdb_family_country.parquet")

print("All data loaded.\n")


# ════════════════════════════════════════════════════════════════════════════
# 1. Global time-series tables
# ════════════════════════════════════════════════════════════════════════════
years_all = np.arange(YEAR_START, YEAR_END + 1)

# Green applications per year
green_yr = (
    green
    .filter((pl.col("yr") >= YEAR_START) & (pl.col("yr") <= YEAR_END))
    .group_by("yr").agg(pl.len().alias("green"))
    .sort("yr").to_pandas().set_index("yr")["green"]
    .reindex(years_all, fill_value=0)
)

# Citation-weighted quality index: sum of forward citations per filing year
cw_idx = (
    green
    .filter((pl.col("yr") >= YEAR_START) & (pl.col("yr") <= YEAR_END))
    .group_by("yr")
    .agg(pl.col("nb_citing_docdb_fam").sum().alias("cw"))
    .sort("yr").to_pandas().set_index("yr")["cw"]
    .reindex(years_all, fill_value=0)
)

total_yr = total_yr_df.reindex(years_all, fill_value=0)
green_share = (green_yr / total_yr * 100).fillna(0)

# Cumulative patent stock (Griliches depreciation approach: δ = 0.15)
delta = 0.15
cum_stock = np.zeros(len(years_all))
for i, y in enumerate(years_all):
    cum_stock[i] = (green_yr.iloc[i]
                    + (cum_stock[i - 1] * (1 - delta) if i > 0 else 0))

# YoY growth rate
yoy_growth = green_yr.pct_change() * 100

# OLS log-linear trend (skip the very first year if outlier)
mask_trend = (years_all >= 1986) & (years_all <= 2021)
log_green = np.log(green_yr.values[mask_trend].astype(float) + 1)
slope, intercept, r, _, _ = stats.linregress(years_all[mask_trend], log_green)
trend_line = np.exp(intercept + slope * years_all)


# ════════════════════════════════════════════════════════════════════════════
# 2. Country-year tables
# ════════════════════════════════════════════════════════════════════════════
green_ctry_yr = (
    green
    .filter((pl.col("yr") >= YEAR_START) & (pl.col("yr") <= YEAR_END))
    .filter(pl.col("appln_auth").is_in(KEY_CTRY))
    .group_by(["yr", "appln_auth"]).agg(pl.len().alias("count"))
    .sort(["yr", "appln_auth"]).to_pandas()
    .pivot(index="yr", columns="appln_auth", values="count")
    .fillna(0).astype(int)
    .reindex(years_all)
    .fillna(0).astype(int)
)

# Green intensity (green / country total granted)
intensity = pd.DataFrame(index=years_all, columns=KEY_CTRY, dtype=float)
for c in KEY_CTRY:
    if c in green_ctry_yr.columns and c in total_ctry_yr_wide.columns:
        g = green_ctry_yr[c].reindex(years_all, fill_value=0)
        t = total_ctry_yr_wide[c].reindex(years_all, fill_value=0)
        intensity[c] = np.where(t > 50, g / t * 100, np.nan)

# Country share of global green patents
green_global_total = green_ctry_yr.sum(axis=1).replace(0, np.nan)
ctry_share = green_ctry_yr.div(green_global_total, axis=0) * 100


# ════════════════════════════════════════════════════════════════════════════
# 3. Cross-country citation flows
# ════════════════════════════════════════════════════════════════════════════
FLOW_CTRY = ["US", "CN", "JP", "DE", "KR", "GB", "FR", "TW", "AU", "CA", "DK", "NL"]

cit_fwd = cit_edges.filter(pl.col("direction") == "forward")
cit_with_ctry = (
    cit_fwd
    .join(fam_ctry.rename({"country": "citing_ctry"}), on="docdb_family_id", how="left")
    .join(
        fam_ctry.rename({"country": "cited_ctry", "docdb_family_id": "cited_docdb_family_id"}),
        on="cited_docdb_family_id", how="left",
    )
    .filter(
        pl.col("citing_ctry").is_in(FLOW_CTRY) & pl.col("cited_ctry").is_in(FLOW_CTRY)
    )
)

# Total bilateral matrix (including same-country)
flow_matrix = (
    cit_with_ctry
    .group_by(["citing_ctry", "cited_ctry"]).agg(pl.len().alias("n"))
    .to_pandas()
    .pivot(index="citing_ctry", columns="cited_ctry", values="n")
    .reindex(index=FLOW_CTRY, columns=FLOW_CTRY)
    .fillna(0).astype(int)
)

# Cross-country only (off-diagonal)
cross_flow = flow_matrix.copy()
np.fill_diagonal(cross_flow.values, 0)

# Domestic vs international ratio
domestic_n = np.diag(flow_matrix.values)
international_n = cross_flow.sum(axis=1).values
dom_int_ratio = pd.Series(
    domestic_n / (domestic_n + international_n + 1e-9) * 100,
    index=FLOW_CTRY,
)

# US–China bilateral over time (approximate using green filing year)
# We don't have year on citation edges, so we use a static snapshot —
# documented as lifetime aggregate in the output


# ════════════════════════════════════════════════════════════════════════════
# helper: shade incomplete-year region
# ════════════════════════════════════════════════════════════════════════════
def shade_incomplete(ax, alpha=0.15, label=True):
    ax.axvspan(INCOMPLETE_FROM - 0.5, YEAR_END + 0.5,
               alpha=alpha, color="#e76f51", zorder=0)
    if label:
        ax.text(INCOMPLETE_FROM + 0.3, ax.get_ylim()[1] * 0.95,
                "Filing lag\n(incomplete)", fontsize=7, color="#c0392b",
                va="top")


# ════════════════════════════════════════════════════════════════════════════
# PLOT A1 – Log-scale trend with OLS fit  (Hall / Griliches style)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(years_all, green_yr.values, color="#2a9d8f", lw=1.8, label="Green applications (log scale)")
ax.semilogy(years_all, trend_line, color="#264653", lw=1.8, ls="--",
            label=f"OLS log-linear trend  (β={slope:.3f}/yr, R²={r**2:.3f})")
ax.fill_between(years_all, green_yr.values, alpha=0.12, color="#2a9d8f")
shade_incomplete(ax)
ax.set_xlim(YEAR_START, YEAR_END)
ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel("Green patent applications (log scale)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
plt.title(
    f"Global Green Patent Applications 1985–2025 — Log-Linear Trend\n"
    f"Annual growth rate ≈ {(np.exp(slope) - 1)*100:.1f}%/yr (1986–2021)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/A1_log_trend.png")
plt.close()
print("Saved A1_log_trend.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT A2 – Year-on-year growth rate
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
colors_bar = ["#2a9d8f" if v >= 0 else "#e76f51" for v in yoy_growth.fillna(0)]
ax.bar(years_all, yoy_growth.reindex(years_all, fill_value=0), color=colors_bar, width=0.8)
ax.axhline(0, color="#333", lw=0.8)
shade_incomplete(ax)
# Annotate key events
events = {
    1997: "Kyoto\nProtocol", 2008: "Global\nFinancial\nCrisis",
    2015: "Paris\nAgreement", 2020: "COVID-19",
}
for yr, label in events.items():
    if yr in yoy_growth.index:
        ax.axvline(yr, color="#888", lw=0.8, ls=":")
        ax.text(yr + 0.2, ax.get_ylim()[1] * 0.85, label, fontsize=7, color="#555")
ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel("YoY growth rate (%)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
plt.title("Year-on-Year Growth Rate of Global Green Patent Applications", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/A2_yoy_growth.png")
plt.close()
print("Saved A2_yoy_growth.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT A3 – Citation-weighted quality index vs. raw count (dual axis)
# ════════════════════════════════════════════════════════════════════════════
# Normalise both to 2000 = 100
base_yr = 2000
base_idx = years_all.tolist().index(base_yr)
raw_idx = green_yr.values / green_yr.values[base_idx] * 100
cw_idx_norm = cw_idx.values / cw_idx.values[base_idx] * 100

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)

ax1.plot(years_all, raw_idx, color="#2a9d8f", lw=2, label="Raw count index (2000=100)")
ax2.plot(years_all, cw_idx_norm, color="#e76f51", lw=2, ls="--",
         label="Citation-weighted index (2000=100)")
ax1.fill_between(years_all, raw_idx, alpha=0.10, color="#2a9d8f")

shade_incomplete(ax1)
ax1.set_xlabel("Application filing year", fontsize=10)
ax1.set_ylabel("Raw count index (2000 = 100)", color="#2a9d8f", fontsize=10)
ax2.set_ylabel("Citation-weighted index (2000 = 100)", color="#e76f51", fontsize=10)
ax1.tick_params(axis="y", colors="#2a9d8f")
ax2.tick_params(axis="y", colors="#e76f51")
ax1.set_xlim(YEAR_START, YEAR_END)

lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=9)
plt.title(
    "Citation-Weighted vs. Raw Patent Count Index (Hall 2005 approach)\n"
    "Citation weight = total forward citations received by family",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/A3_citation_weighted_index.png")
plt.close()
print("Saved A3_citation_weighted_index.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT A4 – Cumulative patent stock & green share
# ════════════════════════════════════════════════════════════════════════════
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax_top.fill_between(years_all, cum_stock / 1e3, alpha=0.25, color="#264653")
ax_top.plot(years_all, cum_stock / 1e3, color="#264653", lw=2,
            label="Cumulative green patent stock (δ=0.15)")
ax_top.set_ylabel("Cumulative stock (thousands)", fontsize=9)
ax_top.legend(fontsize=8)
shade_incomplete(ax_top, label=False)

ax_bot.plot(years_all, green_share.values, color="#2a9d8f", lw=2)
ax_bot.fill_between(years_all, green_share.values, alpha=0.15, color="#2a9d8f")
ax_bot.set_ylabel("Green share of all families (%)", fontsize=9)
ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax_bot.set_xlabel("Application filing year", fontsize=10)
shade_incomplete(ax_bot, label=False)

for ax in (ax_top, ax_bot):
    ax.set_xlim(YEAR_START, YEAR_END)

fig.suptitle("Green Patent Stock Accumulation & Global Green Share (1985–2025)", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/A4_stock_and_share.png")
plt.close()
print("Saved A4_stock_and_share.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B1 – Multi-country green patent counts (absolute)
# ════════════════════════════════════════════════════════════════════════════
PLOT_CTRY = ["US", "CN", "JP", "DE", "KR", "GB", "FR", "DK", "TW", "AU", "CA"]
fig, ax = plt.subplots(figsize=(13, 6))
for c in PLOT_CTRY:
    if c in green_ctry_yr.columns:
        ax.plot(years_all, green_ctry_yr[c].values / 1000,
                color=CTRY_COLOR[c], lw=1.8, label=CTRY_LABEL[c])
shade_incomplete(ax, label=True)
ax.set_xlim(YEAR_START, YEAR_END)
ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel("Green patent applications (thousands)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax.legend(ncol=3, fontsize=8, loc="upper left")
plt.title("Green Patent Applications by Country 1985–2025", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/B1_country_absolute.png")
plt.close()
print("Saved B1_country_absolute.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B2 – Green intensity per country (green / country total granted)
# ════════════════════════════════════════════════════════════════════════════
INTENS_CTRY = ["US", "CN", "JP", "DE", "KR", "GB", "FR", "DK", "AU", "CA"]
fig, ax = plt.subplots(figsize=(13, 6))
for c in INTENS_CTRY:
    ser = intensity[c].dropna()
    if len(ser) > 3:
        ax.plot(ser.index, ser.values, color=CTRY_COLOR[c], lw=1.8,
                label=CTRY_LABEL[c])
shade_incomplete(ax, label=True)
ax.set_xlim(YEAR_START, YEAR_END)
ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel("Green patents / all granted (%) — per country", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.legend(ncol=2, fontsize=8, loc="upper left")
plt.title(
    "Green Patent Intensity by Country 1985–2025\n"
    "(Green applications / total granted applications filed in same office)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/B2_country_intensity.png")
plt.close()
print("Saved B2_country_intensity.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B3 – Country share of global green patents (stacked area)
# ════════════════════════════════════════════════════════════════════════════
STACK_CTRY = ["US", "CN", "JP", "KR", "DE", "DK", "TW", "AU", "CA", "GB", "FR"]
fig, ax = plt.subplots(figsize=(13, 6))
stack_data = [ctry_share[c].reindex(years_all, fill_value=0).values for c in STACK_CTRY if c in ctry_share]
stack_labels = [CTRY_LABEL[c] for c in STACK_CTRY if c in ctry_share]
stack_colors = [CTRY_COLOR[c] for c in STACK_CTRY if c in ctry_share]
ax.stackplot(years_all, *stack_data, labels=stack_labels,
             colors=stack_colors, alpha=0.88)
shade_incomplete(ax, alpha=0.1, label=True)
ax.set_xlim(YEAR_START, YEAR_END)
ax.set_ylim(0, 100)
ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel("Share of tracked-country green patents (%)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(ncol=3, fontsize=7.5, loc="lower left")
plt.title("Country Share of Global Green Patent Filings 1985–2025", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/B3_country_share_stacked.png")
plt.close()
print("Saved B3_country_share_stacked.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B4 – US vs. China: absolute + intensity side by side
# ════════════════════════════════════════════════════════════════════════════
fig, (ax_abs, ax_int) = plt.subplots(1, 2, figsize=(14, 5))

for c, label in [("US", "United States"), ("CN", "China")]:
    ax_abs.plot(years_all, green_ctry_yr[c].values / 1000,
                color=CTRY_COLOR[c], lw=2.2, label=label)
    ax_abs.fill_between(years_all, green_ctry_yr[c].values / 1000,
                        alpha=0.10, color=CTRY_COLOR[c])
    intens = intensity[c].reindex(years_all)
    ax_int.plot(intens.index, intens.values, color=CTRY_COLOR[c], lw=2.2, label=label)

for ax in (ax_abs, ax_int):
    shade_incomplete(ax, label=False)
    ax.set_xlim(YEAR_START, YEAR_END)
    ax.set_xlabel("Application filing year", fontsize=10)
    ax.legend(fontsize=10)

ax_abs.set_ylabel("Green applications (thousands)", fontsize=10)
ax_abs.set_title("Absolute Green Filings", fontsize=10)
ax_int.set_ylabel("Green / total granted (%)", fontsize=10)
ax_int.set_title("Green Patent Intensity", fontsize=10)
ax_int.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax_abs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))

# Annotate convergence
ax_abs.annotate("Convergence\n~2022", xy=(2022, 9), xytext=(2014, 22),
                arrowprops=dict(arrowstyle="->", color="#444"), fontsize=8.5, color="#444")
fig.suptitle("US vs. China Green Patent Comparison 1985–2025", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/B4_us_china_dual.png")
plt.close()
print("Saved B4_us_china_dual.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B5 – Normalised index (2000 = 100) for key countries
# ════════════════════════════════════════════════════════════════════════════
BASE_YR = 2000
NORM_CTRY = ["US", "CN", "JP", "DE", "KR", "DK", "GB"]
fig, ax = plt.subplots(figsize=(13, 5.5))
for c in NORM_CTRY:
    series = green_ctry_yr[c].reindex(years_all, fill_value=0)
    base = series.loc[BASE_YR]
    if base > 0:
        normed = series / base * 100
        ax.plot(years_all, normed.values, color=CTRY_COLOR[c], lw=1.8,
                label=CTRY_LABEL[c])
ax.axhline(100, color="#aaa", lw=0.8, ls="--")
ax.text(YEAR_START + 0.5, 105, f"{BASE_YR} = 100 baseline", fontsize=8, color="#888")
shade_incomplete(ax, label=True)
ax.set_xlim(YEAR_START, YEAR_END)
ax.set_ylim(0)
ax.set_xlabel("Application filing year", fontsize=10)
ax.set_ylabel(f"Green filing index ({BASE_YR} = 100)", fontsize=10)
ax.legend(ncol=2, fontsize=8.5)
plt.title(f"Relative Growth in Green Patenting by Country (index {BASE_YR} = 100)", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/B5_country_index.png")
plt.close()
print("Saved B5_country_index.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C1 – Cross-country citation flow heatmap (log scale)
# ════════════════════════════════════════════════════════════════════════════
flow_log = np.log1p(cross_flow.values)  # log(1 + n) for visual scaling
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(flow_log, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(FLOW_CTRY)))
ax.set_yticks(range(len(FLOW_CTRY)))
ax.set_xticklabels(FLOW_CTRY, fontsize=9)
ax.set_yticklabels(FLOW_CTRY, fontsize=9)
ax.set_xlabel("Cited country (knowledge source)", fontsize=10)
ax.set_ylabel("Citing country (knowledge recipient)", fontsize=10)
# Annotate cells with raw counts (abbreviated)
for i in range(len(FLOW_CTRY)):
    for j in range(len(FLOW_CTRY)):
        n = cross_flow.values[i, j]
        if n > 500:
            txt = f"{n/1000:.0f}k" if n >= 1000 else str(n)
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="black" if flow_log[i, j] < flow_log.max() * 0.65 else "white")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("log(1 + citations)", fontsize=9)
plt.title(
    "Cross-Country Green Patent Citation Flows (lifetime aggregate)\n"
    "Row = citing country · Column = cited country",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/C1_citation_flow_heatmap.png")
plt.close()
print("Saved C1_citation_flow_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C2 – Domestic vs. international citation ratio
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
dom_vals = dom_int_ratio.values
int_vals = 100 - dom_vals
x = np.arange(len(FLOW_CTRY))
ax.bar(x, dom_vals, label="Domestic citations (%)", color="#264653", width=0.65)
ax.bar(x, int_vals, bottom=dom_vals, label="International citations (%)", color="#2a9d8f",
       width=0.65, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(FLOW_CTRY, fontsize=9)
ax.set_ylabel("Share of citations (%)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.axhline(50, color="#aaa", lw=0.7, ls="--")
ax.legend(fontsize=9)
# Annotate with total citation count
for i, c in enumerate(FLOW_CTRY):
    total = domestic_n[i] + international_n[i]
    ax.text(i, 103, f"{total/1000:.0f}k", ha="center", fontsize=7, color="#444")
plt.title(
    "Domestic vs. International Green Patent Citations by Country\n"
    "Values above bars = total citations given (thousands)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/C2_domestic_vs_international.png")
plt.close()
print("Saved C2_domestic_vs_international.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C3 – Net knowledge flows (received − given) per country
# ════════════════════════════════════════════════════════════════════════════
# citations received from abroad − citations given abroad
received = cross_flow.sum(axis=0)   # column sum = citations received
given    = cross_flow.sum(axis=1)   # row sum    = citations given
net_flow = received - given

fig, ax = plt.subplots(figsize=(10, 5))
colors_net = ["#2a9d8f" if v >= 0 else "#e76f51" for v in net_flow.values]
ax.bar(FLOW_CTRY, net_flow.values / 1000, color=colors_net, width=0.65)
ax.axhline(0, color="#333", lw=0.8)
ax.set_ylabel("Net international citations (thousands)\nPositive = net knowledge exporter", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
for i, (c, v) in enumerate(zip(FLOW_CTRY, net_flow.values)):
    ax.text(i, v / 1000 + (2 if v >= 0 else -8), f"{v/1000:.0f}k",
            ha="center", fontsize=8, color="#333")
plt.title(
    "Net International Knowledge Flows via Green Patent Citations\n"
    "(Citations received from abroad  minus  citations given abroad)",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/C3_net_knowledge_flows.png")
plt.close()
print("Saved C3_net_knowledge_flows.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C4 – US–China bilateral citation summary (bar)
# ════════════════════════════════════════════════════════════════════════════
us_to_cn = cross_flow.loc["US", "CN"]
cn_to_us = cross_flow.loc["CN", "US"]
categories = ["CN cites US\n(CN absorbs US knowledge)",
               "US cites CN\n(US absorbs CN knowledge)"]
values = [cn_to_us, us_to_cn]
colors_uc = [CTRY_COLOR["CN"], CTRY_COLOR["US"]]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(categories, [v / 1000 for v in values], color=colors_uc, width=0.45)
ax.set_ylabel("Cross-border citations (thousands)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{v/1000:.0f}k", ha="center", va="bottom", fontsize=11, fontweight="bold")
ratio = cn_to_us / us_to_cn
ax.text(0.5, 0.88, f"Asymmetry ratio: {ratio:.2f}×\n(China cites US green patents {ratio:.1f}× more)", 
        ha="center", transform=ax.transAxes, fontsize=9, color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#ccc"))
plt.title("US–China Green Patent Citation Asymmetry\n(Lifetime aggregate)", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/C4_us_china_bilateral.png")
plt.close()
print("Saved C4_us_china_bilateral.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C5 – Citation-category flow Sankey-style grouped bar
#   H→H, G→H, N→H  /  H→G, G→G, N→G  (who cites whom)
# ════════════════════════════════════════════════════════════════════════════
cat_flow = (
    cit_edges.filter(pl.col("direction") == "forward")
    .group_by(["citing_category", "cited_category"])
    .agg(pl.len().alias("n"))
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(9, 5))
cat_pairs = cat_flow.apply(
    lambda r: f"{r['citing_category']}→{r['cited_category']}", axis=1
)
colors_cat = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#6a4c93","#4a90d9","#81b29a"]
bars = ax.bar(cat_pairs, cat_flow["n"] / 1e6, color=colors_cat[:len(cat_flow)], width=0.65)
ax.set_ylabel("Citation count (millions)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
for bar, v in zip(bars, cat_flow["n"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{v/1e6:.2f}M", ha="center", fontsize=8.5)
ax.set_xlabel("Citing category → Cited category\n(H=High-influence, G=Green, N=Neighbor)", fontsize=9)
plt.title(
    "Citation Flows Between Patent Categories\n"
    "H=High-influence green · G=Green · N=Non-green neighbor",
    fontsize=11, pad=10,
)
fig.tight_layout()
fig.savefig(f"{OUT}/C5_category_flow.png")
plt.close()
print("Saved C5_category_flow.png")


# ════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ════════════════════════════════════════════════════════════════════════════
n_fam_total = green["docdb_family_id"].n_unique()
avg_cites = green["nb_citing_docdb_fam"].mean()
pct_mitigation = green.filter(pl.col("mitigation_adaptation") == "mitigation").height / green.height * 100

print("\n" + "=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)
print(f"  Total green applications (all years)   : {green.height:>12,}")
print(f"  Unique DOCDB families                  : {n_fam_total:>12,}")
print(f"  Avg. forward citations per application : {avg_cites:>12.2f}")
print(f"  Mitigation patents                     : {pct_mitigation:>11.1f}%")
print(f"  OLS log-linear slope (1986-2021)       : {slope:>12.4f}")
print(f"  Implied annual growth rate             : {(np.exp(slope)-1)*100:>11.2f}%")
print(f"  R2 of log-linear fit                   : {r**2:>12.4f}")
print(f"\n  US->CN cross-border citations          : {us_to_cn:>12,}")
print(f"  CN->US cross-border citations          : {cn_to_us:>12,}")
print(f"  Asymmetry ratio (CN->US / US->CN)      : {cn_to_us/us_to_cn:>12.2f}x")
print(f"\n  All figures saved to: {OUT}/")
